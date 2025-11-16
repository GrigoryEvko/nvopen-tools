// Function: sub_2210A20
// Address: 0x2210a20
//
int sub_2210A20()
{
  _QWORD *v0; // rbx
  int result; // eax
  _QWORD *v2; // rax

  v0 = pthread_getspecific(dword_4FD65FC);
  for ( result = pthread_setspecific(dword_4FD65FC, 0); v0; result = ((__int64 (__fastcall *)(_QWORD *))v2[1])(v2) )
  {
    v2 = v0;
    v0 = (_QWORD *)*v0;
  }
  return result;
}
