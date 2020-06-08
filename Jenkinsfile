pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                slackSend color: 'warning',
                                  message: "Build started for branch: ${env.BRANCH_NAME} change: CHANGE_TITLE url: CHANGE_URL"
                echo 'Building..' // Do some work here...
                slackSend color: 'good',
                                  message: "Completed build phase."
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
    post {
        // always {
        //     echo 'One way or another, I have finished'
        //     deleteDir() /* clean up our workspace */
        // }
        success {
            echo 'I succeeeded!'
            slackSend color: 'good',
                      message: "Tsaul good!"
        }
        unstable {
            echo 'I am unstable :/'
        }
        failure {
            echo 'I failed :('
        }
        changed {
            echo 'Things were different before...'
        }
    }
}