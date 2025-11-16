// Function: sub_298F2D0
// Address: 0x298f2d0
//
__int64 __fastcall sub_298F2D0(__int64 a1)
{
  __int64 v2; // rax
  __int64 result; // rax
  const __m128i *v4; // rax
  unsigned __int32 v5; // r13d
  unsigned int v6; // esi
  __int64 v7; // rdx
  int v8; // r14d
  __int64 v9; // rdi
  __int64 *v10; // r8
  unsigned int i; // eax
  __int64 v12; // r9
  __int64 v13; // r11
  unsigned int v14; // eax
  _DWORD *v15; // r8
  __m128i *v16; // rsi
  int v17; // eax
  __int64 v18; // rax
  const __m128i *v19; // rdx
  __int64 v20; // r12
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rcx
  int v24; // r11d
  __int64 v25; // rdi
  __int64 *v26; // r8
  unsigned int j; // eax
  _QWORD *v28; // r9
  __int64 v29; // r15
  int v30; // edx
  __int64 v31; // rax
  unsigned int v32; // eax
  int v33; // ecx
  __int64 v34; // rax
  int v35; // eax
  int v36; // eax
  __int64 v37; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v38; // [rsp+8h] [rbp-58h]
  __int64 *v39; // [rsp+18h] [rbp-48h] BYREF
  __m128i v40; // [rsp+20h] [rbp-40h] BYREF

  v38 = (unsigned __int64 *)(a1 + 64);
  v2 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v2 )
    *(_QWORD *)(a1 + 72) = v2;
  v37 = a1 + 8;
  do
  {
    result = *(_QWORD *)(a1 + 96);
    if ( *(_QWORD *)(a1 + 88) == result )
      return result;
    sub_298EEE0(a1);
    v4 = (const __m128i *)(*(_QWORD *)(a1 + 96) - 96LL);
    v40 = _mm_loadu_si128(v4);
    v5 = v4[5].m128i_u32[2];
    *(_QWORD *)(a1 + 96) = v4;
    if ( *(const __m128i **)(a1 + 88) != v4 && v4[-1].m128i_i32[2] > v5 )
      v4[-1].m128i_i32[2] = v5;
    v6 = *(_DWORD *)(a1 + 32);
    if ( !v6 )
    {
      ++*(_QWORD *)(a1 + 8);
      v39 = 0;
LABEL_50:
      v6 *= 2;
      goto LABEL_51;
    }
    v7 = v40.m128i_i64[0];
    v8 = 1;
    v9 = *(_QWORD *)(a1 + 16);
    v10 = 0;
    for ( i = (v6 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned __int32)v40.m128i_i32[2] >> 9) ^ ((unsigned __int32)v40.m128i_i32[2] >> 4)
                | ((unsigned __int64)(((unsigned __int32)v40.m128i_i32[0] >> 9)
                                    ^ ((unsigned __int32)v40.m128i_i32[0] >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned __int32)v40.m128i_i32[2] >> 9) ^ ((unsigned __int32)v40.m128i_i32[2] >> 4))));
          ;
          i = (v6 - 1) & v14 )
    {
      v12 = v9 + 24LL * i;
      v13 = *(_QWORD *)v12;
      if ( *(_OWORD *)v12 == *(_OWORD *)&v40 )
      {
        v17 = *(_DWORD *)(v12 + 16);
        goto LABEL_23;
      }
      if ( v13 == -4096 )
        break;
      if ( v13 == -8192 && *(_QWORD *)(v12 + 8) == -8192 && !v10 )
        v10 = (__int64 *)(v9 + 24LL * i);
LABEL_16:
      v14 = v8 + i;
      ++v8;
    }
    if ( *(_QWORD *)(v12 + 8) != -4096 )
      goto LABEL_16;
    v36 = *(_DWORD *)(a1 + 24);
    if ( !v10 )
      v10 = (__int64 *)v12;
    ++*(_QWORD *)(a1 + 8);
    v33 = v36 + 1;
    v39 = v10;
    if ( 4 * (v36 + 1) >= 3 * v6 )
      goto LABEL_50;
    if ( v6 - *(_DWORD *)(a1 + 28) - v33 <= v6 >> 3 )
    {
LABEL_51:
      sub_298E6A0(v37, v6);
      sub_298BE50(v37, v40.m128i_i64, &v39);
      v7 = v40.m128i_i64[0];
      v10 = v39;
      v33 = *(_DWORD *)(a1 + 24) + 1;
    }
    *(_DWORD *)(a1 + 24) = v33;
    if ( *v10 != -4096 || v10[1] != -4096 )
      --*(_DWORD *)(a1 + 28);
    *v10 = v7;
    v34 = v40.m128i_i64[1];
    *((_DWORD *)v10 + 4) = 0;
    v10[1] = v34;
    v17 = 0;
LABEL_23:
    ;
  }
  while ( v17 != v5 );
  v16 = *(__m128i **)(a1 + 72);
  while ( 1 )
  {
    v18 = *(_QWORD *)(a1 + 48);
    v19 = (const __m128i *)(v18 - 16);
    if ( *(__m128i **)(a1 + 80) == v16 )
    {
      sub_298BF40(v38, v16, v19);
      v19 = (const __m128i *)(*(_QWORD *)(a1 + 48) - 16LL);
    }
    else
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128((const __m128i *)(v18 - 16));
        v16 = *(__m128i **)(a1 + 72);
        v19 = (const __m128i *)(*(_QWORD *)(a1 + 48) - 16LL);
      }
      *(_QWORD *)(a1 + 72) = v16 + 1;
    }
    v20 = *(_QWORD *)(a1 + 72);
    v21 = *(_DWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 48) = v19;
    if ( !v21 )
    {
      ++*(_QWORD *)(a1 + 8);
      v39 = 0;
LABEL_40:
      v21 *= 2;
      goto LABEL_41;
    }
    v22 = *(_QWORD *)(v20 - 16);
    v23 = *(_QWORD *)(v20 - 8);
    v24 = 1;
    v26 = 0;
    for ( j = (v21 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)
                | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)))); ; j = (v21 - 1) & v32 )
    {
      v25 = *(_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(v25 + 24LL * j);
      v29 = *v28;
      if ( *v28 == v22 && v28[1] == v23 )
      {
        v15 = v28 + 2;
        goto LABEL_19;
      }
      if ( v29 == -4096 )
        break;
      if ( v29 == -8192 && v28[1] == -8192 && !v26 )
        v26 = (__int64 *)(v25 + 24LL * j);
LABEL_48:
      v32 = v24 + j;
      ++v24;
    }
    if ( v28[1] != -4096 )
      goto LABEL_48;
    v35 = *(_DWORD *)(a1 + 24);
    if ( !v26 )
      v26 = v28;
    ++*(_QWORD *)(a1 + 8);
    v30 = v35 + 1;
    v39 = v26;
    if ( 4 * (v35 + 1) >= 3 * v21 )
      goto LABEL_40;
    if ( v21 - *(_DWORD *)(a1 + 28) - v30 <= v21 >> 3 )
    {
LABEL_41:
      sub_298E6A0(v37, v21);
      sub_298BE50(v37, (__int64 *)(v20 - 16), &v39);
      v26 = v39;
      v30 = *(_DWORD *)(a1 + 24) + 1;
    }
    *(_DWORD *)(a1 + 24) = v30;
    if ( *v26 != -4096 || v26[1] != -4096 )
      --*(_DWORD *)(a1 + 28);
    v15 = v26 + 2;
    *((_QWORD *)v15 - 2) = *(_QWORD *)(v20 - 16);
    v31 = *(_QWORD *)(v20 - 8);
    *v15 = 0;
    *((_QWORD *)v15 - 1) = v31;
LABEL_19:
    *v15 = -1;
    v16 = *(__m128i **)(a1 + 72);
    if ( v16[-1].m128i_i64[0] == v40.m128i_i64[0] )
    {
      result = v40.m128i_i64[1];
      if ( v16[-1].m128i_i64[1] == v40.m128i_i64[1] )
        return result;
    }
  }
}
