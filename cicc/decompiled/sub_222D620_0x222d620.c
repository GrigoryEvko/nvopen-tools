// Function: sub_222D620
// Address: 0x222d620
//
int __fastcall sub_222D620(_QWORD *pointer)
{
  void *v2; // rax
  pthread_key_t v3; // edi

  pointer[1] = sub_222D6A0;
  if ( &_pthread_key_create )
    pthread_once(&dword_4FD65E8, sub_2210A70);
  v2 = pthread_getspecific(dword_4FD65FC);
  v3 = dword_4FD65FC;
  *pointer = v2;
  return pthread_setspecific(v3, pointer);
}
