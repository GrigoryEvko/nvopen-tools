// Function: sub_C9E810
// Address: 0xc9e810
//
int __fastcall sub_C9E810(__int64 *a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5, pthread_mutex_t *a6)
{
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 *v11; // rax

  *a1 = (__int64)(a1 + 2);
  sub_C9CA00(a1, a2, (__int64)&a2[a3]);
  a1[4] = (__int64)(a1 + 6);
  sub_C9CA00(a1 + 4, a4, (__int64)&a4[a5]);
  a1[8] = 0;
  a1[9] = 0;
  a1[10] = 0;
  a1[11] = 0;
  if ( &_pthread_key_create )
  {
    v9 = pthread_mutex_lock(a6);
    if ( v9 )
      sub_4264C5(v9);
  }
  v10 = qword_4F84F78;
  if ( qword_4F84F78 )
    *(_QWORD *)(qword_4F84F78 + 96) = a1 + 13;
  a1[13] = v10;
  v11 = &qword_4F84F78;
  a1[12] = (__int64)&qword_4F84F78;
  qword_4F84F78 = (__int64)a1;
  if ( &_pthread_key_create )
    LODWORD(v11) = pthread_mutex_unlock(a6);
  return (int)v11;
}
