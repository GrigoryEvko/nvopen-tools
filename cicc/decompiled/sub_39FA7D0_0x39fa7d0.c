// Function: sub_39FA7D0
// Address: 0x39fa7d0
//
int __fastcall sub_39FA7D0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  int result; // eax

  v1 = (_QWORD *)malloc(0x30u);
  v1[3] = a1;
  v2 = (__int64)v1;
  *v1 = -1;
  v1[1] = 0;
  v1[2] = 0;
  v1[4] = 2042;
  if ( &_pthread_key_create )
    pthread_mutex_lock(&stru_50578C0);
  v3 = qword_50578F8;
  qword_50578F8 = v2;
  *(_QWORD *)(v2 + 40) = v3;
  result = dword_50578E8;
  if ( !dword_50578E8 )
    dword_50578E8 = 1;
  if ( &_pthread_key_create )
    return pthread_mutex_unlock(&stru_50578C0);
  return result;
}
