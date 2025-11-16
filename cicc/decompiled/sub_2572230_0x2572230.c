// Function: sub_2572230
// Address: 0x2572230
//
unsigned __int64 __fastcall sub_2572230(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int32 v4; // eax
  unsigned int v5; // esi
  __int32 v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // r10
  int v10; // r14d
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  int v17; // ecx
  unsigned __int64 result; // rax
  __m128i *v19; // rsi
  int v20; // eax
  int v21; // edx
  __int64 v22; // [rsp+8h] [rbp-48h] BYREF
  __m128i v23; // [rsp+10h] [rbp-40h] BYREF
  __m128i v24; // [rsp+20h] [rbp-30h] BYREF

  v2 = a1 + 8;
  v4 = *(_DWORD *)a1;
  v22 = a2;
  v5 = *(_DWORD *)(a1 + 32);
  v6 = v4 + 1;
  *(_DWORD *)a1 = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    v23.m128i_i64[0] = 0;
    goto LABEL_31;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = v22;
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v12 = v7 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v22 != *(_QWORD *)v12 )
  {
    while ( v13 != -4096 )
    {
      if ( !v9 && v13 == -8192 )
        v9 = v12;
      v11 = (v5 - 1) & (v10 + v11);
      v12 = v7 + 16LL * v11;
      v13 = *(_QWORD *)v12;
      if ( v22 == *(_QWORD *)v12 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v9 )
      v9 = v12;
    v20 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v21 = v20 + 1;
    v23.m128i_i64[0] = v9;
    if ( 4 * (v20 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 28) - v21 > v5 >> 3 )
        goto LABEL_25;
      goto LABEL_32;
    }
LABEL_31:
    v5 *= 2;
LABEL_32:
    sub_B23080(v2, v5);
    sub_B1C700(v2, &v22, &v23);
    v8 = v22;
    v9 = v23.m128i_i64[0];
    v21 = *(_DWORD *)(a1 + 24) + 1;
LABEL_25:
    *(_DWORD *)(a1 + 24) = v21;
    if ( *(_QWORD *)v9 != -4096 )
      --*(_DWORD *)(a1 + 28);
    *(_QWORD *)v9 = v8;
    *(_DWORD *)(v9 + 8) = 0;
    *(_DWORD *)(v9 + 8) = v6;
    v14 = *(_BYTE **)(a1 + 48);
    if ( v14 != *(_BYTE **)(a1 + 56) )
      goto LABEL_4;
LABEL_28:
    sub_9319A0(a1 + 40, v14, &v22);
    v15 = v22;
    goto LABEL_7;
  }
LABEL_3:
  *(_DWORD *)(v12 + 8) = v6;
  v14 = *(_BYTE **)(a1 + 48);
  if ( v14 == *(_BYTE **)(a1 + 56) )
    goto LABEL_28;
LABEL_4:
  v15 = v22;
  if ( v14 )
  {
    *(_QWORD *)v14 = v22;
    v14 = *(_BYTE **)(a1 + 48);
  }
  *(_QWORD *)(a1 + 48) = v14 + 8;
LABEL_7:
  v16 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == v15 + 48 )
  {
    result = 0;
  }
  else
  {
    if ( !v16 )
      BUG();
    v17 = *(unsigned __int8 *)(v16 - 24);
    result = v16 - 24;
    if ( (unsigned int)(v17 - 30) >= 0xB )
      result = 0;
  }
  v24.m128i_i32[2] = *(_DWORD *)a1;
  v19 = *(__m128i **)(a1 + 96);
  v23.m128i_i64[0] = v15;
  v23.m128i_i64[1] = result;
  v24.m128i_i32[0] = 0;
  if ( v19 == *(__m128i **)(a1 + 104) )
    return sub_25720A0((unsigned __int64 *)(a1 + 88), v19, &v23);
  if ( v19 )
  {
    *v19 = _mm_loadu_si128(&v23);
    v19[1] = _mm_loadu_si128(&v24);
    v19 = *(__m128i **)(a1 + 96);
  }
  *(_QWORD *)(a1 + 96) = v19 + 2;
  return result;
}
