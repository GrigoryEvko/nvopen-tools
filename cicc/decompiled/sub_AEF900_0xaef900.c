// Function: sub_AEF900
// Address: 0xaef900
//
void __fastcall sub_AEF900(__int64 a1)
{
  __m128i *v1; // rbx
  __m128i *v2; // r14
  unsigned int v4; // r13d
  __int64 v5; // r15
  unsigned int v6; // r13d
  int v7; // eax
  int v8; // esi
  __int64 *v9; // rcx
  unsigned int j; // edx
  __int64 *v11; // rax
  __int64 v12; // rdi
  int v13; // r13d
  int v14; // eax
  unsigned int v15; // edx
  __int64 v16; // r15
  int v17; // r13d
  int v18; // eax
  int v19; // r8d
  __int64 *v20; // rdi
  unsigned int i; // eax
  __int64 v22; // rdx
  unsigned int v23; // eax
  int v24; // eax
  __int64 *v25; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(__m128i **)(a1 + 32);
  v2 = &v1[*(unsigned int *)(a1 + 40)];
  if ( v2 == v1 )
    return;
  do
  {
    v4 = *(_DWORD *)(a1 + 24);
    if ( !v4 )
    {
      ++*(_QWORD *)a1;
      v25 = 0;
LABEL_12:
      sub_AEF630(a1, 2 * v4);
      v13 = *(_DWORD *)(a1 + 24);
      if ( v13 )
      {
        v16 = *(_QWORD *)(a1 + 8);
        v17 = v13 - 1;
        v18 = sub_AEA4A0(v1->m128i_i64, &v1->m128i_i64[1]);
        v19 = 1;
        v20 = 0;
        for ( i = v17 & v18; ; i = v17 & v23 )
        {
          v9 = (__int64 *)(v16 + 16LL * i);
          v22 = *v9;
          if ( *v9 == v1->m128i_i64[0] && v1->m128i_i64[1] == v9[1] )
            break;
          if ( v22 == -4096 )
          {
            if ( v9[1] == -4096 )
            {
              if ( v20 )
                v9 = v20;
              break;
            }
          }
          else if ( v22 == -8192 && v9[1] == -8192 && !v20 )
          {
            v20 = (__int64 *)(v16 + 16LL * i);
          }
          v23 = v19 + i;
          ++v19;
        }
        v25 = v9;
      }
      else
      {
        v25 = 0;
        v9 = 0;
      }
      v14 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_15;
    }
    v5 = *(_QWORD *)(a1 + 8);
    v6 = v4 - 1;
    v7 = sub_AEA4A0(v1->m128i_i64, &v1->m128i_i64[1]);
    v8 = 1;
    v9 = 0;
    for ( j = v6 & v7; ; j = v6 & v15 )
    {
      v11 = (__int64 *)(v5 + 16LL * j);
      v12 = *v11;
      if ( *v11 == v1->m128i_i64[0] )
        break;
      if ( v12 == -4096 )
        goto LABEL_22;
LABEL_6:
      if ( v12 == -8192 && v11[1] == -8192 && !v9 )
        v9 = (__int64 *)(v5 + 16LL * j);
LABEL_23:
      v15 = v8 + j;
      ++v8;
    }
    if ( v1->m128i_i64[1] == v11[1] )
      goto LABEL_18;
    if ( v12 != -4096 )
      goto LABEL_6;
LABEL_22:
    if ( v11[1] != -4096 )
      goto LABEL_23;
    v4 = *(_DWORD *)(a1 + 24);
    if ( !v9 )
      v9 = (__int64 *)(v5 + 16LL * j);
    v24 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v25 = v9;
    v14 = v24 + 1;
    if ( 4 * v14 >= 3 * v4 )
      goto LABEL_12;
    if ( v4 - (v14 + *(_DWORD *)(a1 + 20)) <= v4 >> 3 )
    {
      sub_AEF630(a1, v4);
      sub_AEAAD0(a1, v1->m128i_i64, &v25);
      v9 = v25;
      v14 = *(_DWORD *)(a1 + 16) + 1;
    }
LABEL_15:
    *(_DWORD *)(a1 + 16) = v14;
    if ( *v9 != -4096 || v9[1] != -4096 )
      --*(_DWORD *)(a1 + 20);
    *(__m128i *)v9 = _mm_loadu_si128(v1);
LABEL_18:
    ++v1;
  }
  while ( v2 != v1 );
}
