// Function: sub_23CD060
// Address: 0x23cd060
//
void __fastcall sub_23CD060(__int64 a1)
{
  unsigned int v2; // eax
  int v3; // eax
  pthread_t *v4; // rbx
  pthread_t *i; // r15
  pthread_t v6; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi

  *(_QWORD *)a1 = &unk_4A162C8;
  if ( &_pthread_key_create )
  {
    v2 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 176));
    if ( v2 )
      sub_4264C5(v2);
  }
  *(_BYTE *)(a1 + 352) = 0;
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 176));
  sub_2210B70((pthread_cond_t *)(a1 + 216));
  while ( &_pthread_key_create )
  {
    v3 = pthread_rwlock_rdlock((pthread_rwlock_t *)(a1 + 32));
    if ( v3 != 11 )
    {
      if ( v3 == 35 )
        sub_4264C5(0x23u);
      break;
    }
  }
  v4 = *(pthread_t **)(a1 + 16);
  for ( i = *(pthread_t **)(a1 + 8); v4 != i; *(i - 1) = 0 )
  {
    v6 = *i++;
    sub_C959A0(v6);
  }
  if ( &_pthread_key_create )
    pthread_rwlock_unlock((pthread_rwlock_t *)(a1 + 32));
  sub_C7D6A0(*(_QWORD *)(a1 + 328), 16LL * *(unsigned int *)(a1 + 344), 8);
  j__pthread_cond_destroy((pthread_cond_t *)(a1 + 264));
  j__pthread_cond_destroy((pthread_cond_t *)(a1 + 216));
  sub_23CCEB0((unsigned __int64 *)(a1 + 96));
  v7 = *(_QWORD **)(a1 + 16);
  v8 = *(_QWORD **)(a1 + 8);
  if ( v7 != v8 )
  {
    do
    {
      if ( *v8 )
        sub_2207530();
      ++v8;
    }
    while ( v7 != v8 );
    v8 = *(_QWORD **)(a1 + 8);
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
  nullsub_1496();
}
