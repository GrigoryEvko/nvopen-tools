// Function: sub_37DCA30
// Address: 0x37dca30
//
__int64 __fastcall sub_37DCA30(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, unsigned __int64 a6)
{
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // rdi
  int v11; // r14d
  unsigned int i; // eax
  const __m128i *v13; // r10
  unsigned int v14; // eax
  __m128i v16; // rax
  unsigned int v17; // esi
  __m128i *v18; // rax
  int v19; // ebx
  int v20; // edx
  __m128i *v21; // [rsp+10h] [rbp-70h] BYREF
  __m128i *v22; // [rsp+18h] [rbp-68h] BYREF
  __m128i v23; // [rsp+20h] [rbp-60h] BYREF
  __int128 v24; // [rsp+30h] [rbp-50h] BYREF
  __m128i v25[4]; // [rsp+40h] [rbp-40h] BYREF

  v8 = a6;
  v9 = *(_QWORD *)(a1 + 2144);
  v10 = *(unsigned int *)(a1 + 2160);
  if ( !(_DWORD)v10 )
    goto LABEL_11;
  v11 = 1;
  for ( i = (v10 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * a6) | ((unsigned __int64)(((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4)) << 32))) >> 31)
           ^ (756364221 * a6)); ; i = (v10 - 1) & v14 )
  {
    v13 = (const __m128i *)(v9 + 32LL * i);
    if ( a5 == v13->m128i_i64[0] && (_DWORD)a6 == v13->m128i_i32[2] )
      break;
    if ( v13->m128i_i64[0] == -4096 && v13->m128i_i32[2] == -1 )
      goto LABEL_11;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == (const __m128i *)(v9 + 32 * v10) )
  {
LABEL_11:
    v16.m128i_i64[0] = sub_37CB820(a1, a2, a3, a4, a5, a6);
    *(_QWORD *)&v24 = a5;
    v23 = v16;
    DWORD2(v24) = v8;
    v25[0] = v16;
    if ( (unsigned __int8)sub_37C41D0(a1 + 2136, (__int64 *)&v24, &v21) )
      return _mm_loadu_si128(&v23).m128i_i64[0];
    v17 = *(_DWORD *)(a1 + 2160);
    v18 = v21;
    v19 = *(_DWORD *)(a1 + 2152);
    ++*(_QWORD *)(a1 + 2136);
    v22 = v18;
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) >= 3 * v17 )
    {
      v17 *= 2;
    }
    else if ( v17 - *(_DWORD *)(a1 + 2156) - v20 > v17 >> 3 )
    {
      goto LABEL_15;
    }
    sub_37DC7B0(a1 + 2136, v17);
    sub_37C41D0(a1 + 2136, (__int64 *)&v24, &v22);
    v20 = *(_DWORD *)(a1 + 2152) + 1;
    v18 = v22;
LABEL_15:
    *(_DWORD *)(a1 + 2152) = v20;
    if ( v18->m128i_i64[0] != -4096 || v18->m128i_i32[2] != -1 )
      --*(_DWORD *)(a1 + 2156);
    v18->m128i_i64[0] = v24;
    v18->m128i_i32[2] = DWORD2(v24);
    v18[1] = _mm_loadu_si128(v25);
    return _mm_loadu_si128(&v23).m128i_i64[0];
  }
  return _mm_loadu_si128(v13 + 1).m128i_i64[0];
}
