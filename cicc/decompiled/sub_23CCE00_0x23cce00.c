// Function: sub_23CCE00
// Address: 0x23cce00
//
__int64 __fastcall sub_23CCE00(__int64 a1)
{
  pthread_rwlock_t *v1; // r15
  int v2; // eax
  pthread_t v3; // rax
  __int64 *v4; // rbx
  __int64 *v5; // r13
  pthread_t v6; // r12
  unsigned int v7; // r12d

  v1 = (pthread_rwlock_t *)(a1 + 32);
  while ( &_pthread_key_create )
  {
    v2 = pthread_rwlock_rdlock(v1);
    if ( v2 != 11 )
    {
      if ( v2 == 35 )
        sub_4264C5(0x23u);
      break;
    }
  }
  v3 = j__pthread_self();
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(__int64 **)(a1 + 16);
  v6 = v3;
  if ( v4 == v5 )
  {
LABEL_10:
    v7 = 0;
  }
  else
  {
    while ( v6 != sub_C959C0(*v4) )
    {
      if ( v5 == ++v4 )
        goto LABEL_10;
    }
    v7 = 1;
  }
  if ( &_pthread_key_create )
    pthread_rwlock_unlock(v1);
  return v7;
}
