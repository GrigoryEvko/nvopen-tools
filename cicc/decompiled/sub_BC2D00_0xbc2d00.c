// Function: sub_BC2D00
// Address: 0xbc2d00
//
__int64 __fastcall sub_BC2D00(pthread_rwlock_t *rwlock, __int64 a2, __int64 a3)
{
  int v4; // eax
  unsigned int v5; // eax
  int v6; // eax
  unsigned __int64 pad2; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // r12

  while ( &_pthread_key_create )
  {
    v4 = pthread_rwlock_rdlock(rwlock);
    if ( v4 != 11 )
    {
      if ( v4 == 35 )
        sub_4264C5(0x23u);
      break;
    }
  }
  v5 = sub_C92610(a2, a3);
  v6 = sub_C92860(&rwlock[1].__align + 5, a2, a3, v5);
  if ( v6 == -1 || (pad2 = rwlock[1].__pad2, v8 = pad2 + 8LL * v6, v8 == pad2 + 8LL * rwlock[1].__flags) )
    v9 = 0;
  else
    v9 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
  if ( &_pthread_key_create )
    pthread_rwlock_unlock(rwlock);
  return v9;
}
