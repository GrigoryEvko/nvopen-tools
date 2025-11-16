// Function: sub_16D56E0
// Address: 0x16d56e0
//
__int64 __fastcall sub_16D56E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  pthread_cond_t *v3; // r13
  _QWORD *v4; // r12
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi
  __int64 *v7; // r14
  __int64 **v8; // r13
  __int64 *v9; // r12
  __int64 *v10; // r15
  __int64 v11; // rbx
  __int64 *v12; // rdi
  __int64 v13; // rdx
  __int64 *v14; // rdi
  __int64 *v15; // rdi
  __int64 v16; // rdi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  __int64 result; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rdi
  __int64 i; // [rsp+8h] [rbp-58h]
  __int64 *v24; // [rsp+10h] [rbp-50h]
  unsigned __int64 v26; // [rsp+20h] [rbp-40h]
  __int64 *v27; // [rsp+28h] [rbp-38h]

  if ( &_pthread_key_create )
  {
    v2 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 104));
    if ( v2 )
      sub_4264C5(v2);
  }
  *(_BYTE *)(a1 + 284) = 0;
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 104));
  v3 = (pthread_cond_t *)(a1 + 144);
  sub_2210B70(a1 + 144);
  v4 = *(_QWORD **)(a1 + 8);
  v5 = *(_QWORD **)a1;
  if ( *(_QWORD **)a1 != v4 )
  {
    do
    {
      v6 = v5++;
      sub_2242090(v6);
    }
    while ( v4 != v5 );
  }
  j__pthread_cond_destroy((pthread_cond_t *)(a1 + 232));
  j__pthread_cond_destroy(v3);
  v7 = *(__int64 **)(a1 + 80);
  v27 = *(__int64 **)(a1 + 72);
  v8 = (__int64 **)(*(_QWORD *)(a1 + 64) + 8LL);
  v24 = *(__int64 **)(a1 + 56);
  v9 = *(__int64 **)(a1 + 40);
  v26 = *(_QWORD *)(a1 + 96);
  for ( i = *(_QWORD *)(a1 + 64); v26 > (unsigned __int64)v8; ++v8 )
  {
    v10 = *v8;
    v11 = (__int64)(*v8 + 64);
    do
    {
      v12 = v10;
      v10 += 2;
      sub_16D4F80(v12);
    }
    while ( (__int64 *)v11 != v10 );
  }
  v13 = i;
  if ( v26 == i )
  {
    while ( v27 != v9 )
    {
      v22 = v9;
      v9 += 2;
      sub_16D4F80(v22);
    }
  }
  else
  {
    while ( v24 != v9 )
    {
      v14 = v9;
      v9 += 2;
      sub_16D4F80(v14);
    }
    while ( v27 != v7 )
    {
      v15 = v7;
      v7 += 2;
      sub_16D4F80(v15);
    }
  }
  v16 = *(_QWORD *)(a1 + 24);
  if ( v16 )
  {
    v17 = *(__int64 **)(a1 + 64);
    v18 = *(_QWORD *)(a1 + 96) + 8LL;
    if ( v18 > (unsigned __int64)v17 )
    {
      do
      {
        v19 = *v17++;
        j_j___libc_free_0(v19, 512);
      }
      while ( v18 > (unsigned __int64)v17 );
      v16 = *(_QWORD *)(a1 + 24);
    }
    a2 = 8LL * *(_QWORD *)(a1 + 32);
    j_j___libc_free_0(v16, a2);
  }
  result = *(_QWORD *)(a1 + 8);
  v21 = *(_QWORD **)a1;
  if ( result != *(_QWORD *)a1 )
  {
    do
    {
      if ( *v21 )
        sub_2207530(v21, a2, v13);
      ++v21;
    }
    while ( (_QWORD *)result != v21 );
    result = a1;
    v21 = *(_QWORD **)a1;
  }
  if ( v21 )
    return j_j___libc_free_0(v21, *(_QWORD *)(a1 + 16) - (_QWORD)v21);
  return result;
}
