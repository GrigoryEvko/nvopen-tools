// Function: sub_C9F930
// Address: 0xc9f930
//
__int64 __fastcall sub_C9F930(__int64 a1)
{
  const __m128i *i; // rsi
  pthread_mutex_t *v3; // r12
  unsigned int v4; // eax
  __int64 v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  _QWORD *v9; // rdi
  __int64 v10; // rdi
  __int64 result; // rax

  for ( i = *(const __m128i **)(a1 + 64); i; i = *(const __m128i **)(a1 + 64) )
    sub_C9F770(a1, i);
  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  v3 = (pthread_mutex_t *)(qword_4F84F60 + 664);
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)(qword_4F84F60 + 664));
    if ( v4 )
      sub_4264C5(v4);
  }
  v5 = *(_QWORD *)(a1 + 104);
  **(_QWORD **)(a1 + 96) = v5;
  if ( v5 )
    *(_QWORD *)(v5 + 96) = *(_QWORD *)(a1 + 96);
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v3);
  v6 = *(_QWORD **)(a1 + 80);
  v7 = *(_QWORD **)(a1 + 72);
  if ( v6 != v7 )
  {
    do
    {
      v8 = (_QWORD *)v7[9];
      if ( v8 != v7 + 11 )
        j_j___libc_free_0(v8, v7[11] + 1LL);
      v9 = (_QWORD *)v7[5];
      if ( v9 != v7 + 7 )
        j_j___libc_free_0(v9, v7[7] + 1LL);
      v7 += 13;
    }
    while ( v6 != v7 );
    v7 = *(_QWORD **)(a1 + 72);
  }
  if ( v7 )
    j_j___libc_free_0(v7, *(_QWORD *)(a1 + 88) - (_QWORD)v7);
  v10 = *(_QWORD *)(a1 + 32);
  if ( v10 != a1 + 48 )
    j_j___libc_free_0(v10, *(_QWORD *)(a1 + 48) + 1LL);
  result = a1 + 16;
  if ( *(_QWORD *)a1 != a1 + 16 )
    return j_j___libc_free_0(*(_QWORD *)a1, *(_QWORD *)(a1 + 16) + 1LL);
  return result;
}
