// Function: sub_B07720
// Address: 0xb07720
//
const __m128i *__fastcall sub_B07720(const __m128i *a1, int a2, __int64 a3)
{
  const __m128i *result; // rax
  unsigned int v4; // r13d
  int v6; // esi
  const __m128i **v7; // rcx
  const __m128i *v8; // r12
  int v9; // eax
  __int64 v10; // r14
  int v11; // eax
  unsigned int v12; // r13d
  unsigned int v13; // edx
  const __m128i **v14; // rsi
  int v15; // edi
  const __m128i **v16; // r8
  int v17; // eax
  const __m128i *v18; // [rsp+8h] [rbp-78h] BYREF
  int v19; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v20; // [rsp+18h] [rbp-68h] BYREF
  const __m128i **v21; // [rsp+20h] [rbp-60h] BYREF
  __int64 v22; // [rsp+28h] [rbp-58h] BYREF
  __m128i v23; // [rsp+30h] [rbp-50h]
  __int64 v24; // [rsp+40h] [rbp-40h]
  __int64 v25[7]; // [rsp+48h] [rbp-38h] BYREF

  v18 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v18;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v21 = 0;
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v21 = (const __m128i **)sub_AF5140((__int64)a1, 0);
  v22 = sub_AF5140((__int64)a1, 1u);
  v23 = _mm_loadu_si128(a1 + 1);
  v24 = a1[2].m128i_i64[0];
  v25[0] = a1[2].m128i_i64[1];
  if ( (_BYTE)v24 )
  {
    v20 = v23.m128i_i64[1];
    v11 = v23.m128i_i32[0];
  }
  else
  {
    v20 = 0;
    v11 = 0;
  }
  v19 = v11;
  v12 = v4 - 1;
  v8 = v18;
  v13 = v12 & sub_AFAA60((__int64 *)&v21, &v22, &v19, &v20, v25);
  v14 = (const __m128i **)(v10 + 8LL * v13);
  result = *v14;
  if ( v18 == *v14 )
    return result;
  v15 = 1;
  v7 = 0;
  while ( result != (const __m128i *)-4096LL )
  {
    if ( v7 || result != (const __m128i *)-8192LL )
      v14 = v7;
    v13 = v12 & (v15 + v13);
    v16 = (const __m128i **)(v10 + 8LL * v13);
    result = *v16;
    if ( v18 == *v16 )
      return result;
    ++v15;
    v7 = v14;
    v14 = (const __m128i **)(v10 + 8LL * v13);
  }
  v17 = *(_DWORD *)(a3 + 16);
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v7 )
    v7 = v14;
  ++*(_QWORD *)a3;
  v9 = v17 + 1;
  v21 = v7;
  if ( 4 * v9 >= 3 * v4 )
    goto LABEL_7;
  if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
    goto LABEL_9;
  v6 = v4;
LABEL_8:
  sub_B07440(a3, v6);
  sub_AFDA30(a3, &v18, &v21);
  v7 = v21;
  v8 = v18;
  v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
  *(_DWORD *)(a3 + 16) = v9;
  if ( *v7 != (const __m128i *)-4096LL )
    --*(_DWORD *)(a3 + 20);
  *v7 = v8;
  return v18;
}
