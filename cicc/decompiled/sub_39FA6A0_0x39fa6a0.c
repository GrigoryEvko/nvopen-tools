// Function: sub_39FA6A0
// Address: 0x39fa6a0
//
int __fastcall sub_39FA6A0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  int result; // eax

  *a2 = -1;
  a2[1] = a3;
  a2[2] = a4;
  a2[3] = a1;
  a2[4] = 2042;
  if ( &_pthread_key_create )
    pthread_mutex_lock(&stru_50578C0);
  v4 = qword_50578F8;
  qword_50578F8 = (__int64)a2;
  a2[5] = v4;
  result = dword_50578E8;
  if ( !dword_50578E8 )
    dword_50578E8 = 1;
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(&stru_50578C0);
  return result;
}
