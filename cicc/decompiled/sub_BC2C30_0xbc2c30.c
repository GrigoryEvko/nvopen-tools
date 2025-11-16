// Function: sub_BC2C30
// Address: 0xbc2c30
//
__int64 __fastcall sub_BC2C30(pthread_rwlock_t *rwlock, __int64 a2)
{
  int v4; // eax
  int v5; // eax
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  int v13; // eax
  int v14; // r8d

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
  v5 = *((_DWORD *)&rwlock[1].__align + 8);
  v6 = *(&rwlock[1].__align + 2);
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_6:
      v11 = v9[1];
      goto LABEL_7;
    }
    v13 = 1;
    while ( v10 != -4096 )
    {
      v14 = v13 + 1;
      v8 = v7 & (v13 + v8);
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_6;
      v13 = v14;
    }
  }
  v11 = 0;
LABEL_7:
  if ( &_pthread_key_create )
    pthread_rwlock_unlock(rwlock);
  return v11;
}
