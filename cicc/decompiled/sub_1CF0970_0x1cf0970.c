// Function: sub_1CF0970
// Address: 0x1cf0970
//
__int64 __fastcall sub_1CF0970(__int64 a1, _QWORD *a2, __int64 *a3, __int64 a4, int a5)
{
  unsigned int v6; // r15d
  unsigned int v8; // ebx
  __int64 v9; // rax
  __m128i *v10; // rax
  __m128i v11; // xmm2
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  const __m128i *v15; // rax
  __m128i v16; // xmm0
  unsigned __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rcx
  bool v20; // cc
  const __m128i *v21; // rsi
  __int64 v22; // rcx
  unsigned int v23; // edi
  __m128i v24; // xmm0
  __int64 v25; // rax
  __int64 result; // rax
  _QWORD *v28; // [rsp+18h] [rbp-488h]
  _QWORD *v29; // [rsp+18h] [rbp-488h]
  unsigned __int8 v30; // [rsp+18h] [rbp-488h]
  __m128i v31; // [rsp+20h] [rbp-480h] BYREF
  __m128i v32; // [rsp+30h] [rbp-470h] BYREF
  __m128i v33; // [rsp+40h] [rbp-460h]
  __int64 v34; // [rsp+50h] [rbp-450h]
  _BYTE *v35; // [rsp+60h] [rbp-440h] BYREF
  __int64 v36; // [rsp+68h] [rbp-438h]
  _BYTE v37[1072]; // [rsp+70h] [rbp-430h] BYREF

  v6 = 0;
  v8 = 0;
  v35 = v37;
  v36 = 0x2000000000LL;
  while ( 1 )
  {
    v12 = *a2;
    v13 = a3[1];
    v14 = *a3;
    if ( v6 >= -1431655765 * (unsigned int)((__int64)(a2[1] - *a2) >> 3) )
      break;
    v31 = 0u;
    v32.m128i_i8[8] = 0;
    v21 = (const __m128i *)(v12 + 24LL * v6);
    v32.m128i_i32[0] = 0;
    if ( v8 < -1431655765 * (unsigned int)((v13 - v14) >> 3) )
    {
      v15 = (const __m128i *)(v14 + 24LL * v8);
      v22 = v21->m128i_i64[1];
      v23 = *(_DWORD *)(v15->m128i_i64[1] + 48);
      if ( *(_DWORD *)(v22 + 48) == v23 )
      {
        if ( v21[1].m128i_i32[0] >= (unsigned __int32)v15[1].m128i_i32[0] )
          goto LABEL_10;
      }
      else if ( *(_DWORD *)(v22 + 48) >= v23 )
      {
        goto LABEL_10;
      }
    }
    v24 = _mm_loadu_si128(v21);
    v25 = v21[1].m128i_i64[0];
    ++v6;
    v32.m128i_i8[8] = 1;
    v34 = v25;
    v32.m128i_i64[0] = v25;
    v33 = v24;
    v31 = v24;
LABEL_11:
    v9 = (unsigned int)v36;
    if ( !(_DWORD)v36 )
      goto LABEL_5;
    v17 = (unsigned __int64)&v35[32 * (unsigned int)v36 - 32];
    v18 = *(_DWORD *)(v31.m128i_i64[1] + 48);
    v19 = *(_QWORD *)(v17 + 8);
    v20 = *(_DWORD *)(v19 + 48) <= v18;
    if ( *(_DWORD *)(v19 + 48) == v18 )
    {
LABEL_16:
      if ( *(_DWORD *)(v17 + 16) <= v32.m128i_i32[0] )
        goto LABEL_3;
      goto LABEL_14;
    }
    while ( !v20 || *(_DWORD *)(v31.m128i_i64[1] + 52) > *(_DWORD *)(v19 + 52) )
    {
LABEL_14:
      v9 = (unsigned int)(v9 - 1);
      v17 -= 32LL;
      LODWORD(v36) = v9;
      if ( !(_DWORD)v9 )
        goto LABEL_5;
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)(v19 + 48) <= v18;
      if ( *(_DWORD *)(v19 + 48) == v18 )
        goto LABEL_16;
    }
LABEL_3:
    if ( *(_BYTE *)(v17 + 24) == v32.m128i_i8[8] )
    {
      v9 = (unsigned int)v36;
LABEL_5:
      if ( HIDWORD(v36) > (unsigned int)v9 )
        goto LABEL_6;
LABEL_24:
      v29 = a2;
      sub_16CD150((__int64)&v35, v37, 0, 32, a5, (int)a2);
      v9 = (unsigned int)v36;
      a2 = v29;
      goto LABEL_6;
    }
    v28 = a2;
    result = sub_1CF02B0(a1, v31.m128i_i64, (_QWORD *)v17);
    if ( (_BYTE)result )
      goto LABEL_28;
    v9 = (unsigned int)v36;
    a2 = v28;
    if ( HIDWORD(v36) <= (unsigned int)v36 )
      goto LABEL_24;
LABEL_6:
    v10 = (__m128i *)&v35[32 * v9];
    *v10 = _mm_loadu_si128(&v31);
    v11 = _mm_loadu_si128(&v32);
    LODWORD(v36) = v36 + 1;
    v10[1] = v11;
  }
  if ( v8 < -1431655765 * (unsigned int)((v13 - v14) >> 3) )
  {
    v31 = 0u;
    v32.m128i_i32[0] = 0;
    v15 = (const __m128i *)(v14 + 24LL * v8);
    v32.m128i_i8[8] = 0;
LABEL_10:
    v16 = _mm_loadu_si128(v15);
    ++v8;
    v34 = v15[1].m128i_i64[0];
    v32.m128i_i64[0] = v34;
    v33 = v16;
    v31 = v16;
    goto LABEL_11;
  }
  result = 0;
LABEL_28:
  if ( v35 != v37 )
  {
    v30 = result;
    _libc_free((unsigned __int64)v35);
    return v30;
  }
  return result;
}
