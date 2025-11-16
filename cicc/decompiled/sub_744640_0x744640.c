// Function: sub_744640
// Address: 0x744640
//
__int64 __fastcall sub_744640(
        __int64 a1,
        __m128i *a2,
        int a3,
        __m128i *a4,
        __int64 a5,
        _DWORD *a6,
        int a7,
        int *a8,
        __m128i *a9,
        __m128i *a10)
{
  int v13; // r14d
  __int64 v14; // rdi
  _QWORD *v15; // r12
  __m128i *v16; // r11
  int v17; // eax
  int v18; // r10d
  _BYTE *v19; // rax
  __int64 v20; // rax
  const __m128i *v21; // r15
  __m128i *v22; // rax
  __int64 v23; // r14
  __int8 v24; // al
  int v25; // eax
  _QWORD *v26; // rax
  __int64 result; // rax
  __int64 v28; // rdx
  __m128i *v29; // [rsp+8h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-68h]
  __m128i *v32; // [rsp+20h] [rbp-58h]
  int v34; // [rsp+3Ch] [rbp-3Ch] BYREF
  _QWORD v35[7]; // [rsp+40h] [rbp-38h] BYREF

  v13 = a7;
  v14 = *(_QWORD *)(a1 + 128);
  v34 = 0;
  v15 = (_QWORD *)sub_8A2270(v14, (_DWORD)a4, a5, (_DWORD)a6, a7, (_DWORD)a8, (__int64)a9);
  v16 = a4;
  if ( *a8 )
    return a1;
  v17 = a7;
  if ( (a7 & 0x80) != 0 )
  {
    LOBYTE(v17) = a7 & 0x7F;
    v13 = v17;
  }
  v18 = a3 == 0 ? 4096 : 36864;
  if ( a2[10].m128i_i8[13] == 6 && !a2[11].m128i_i8[0] )
  {
    v19 = (_BYTE *)a2[11].m128i_i64[1];
    if ( (v19[89] & 4) != 0 )
    {
      v20 = sub_8A1CE0(
              *(_QWORD *)v19,
              *(_QWORD *)(*(_QWORD *)v19 + 64LL),
              (_DWORD)a4,
              a5,
              (_DWORD)a6,
              0,
              0,
              v13,
              (__int64)a8,
              (__int64)a9);
      v30 = v20;
      if ( *a8 )
        return a1;
      v16 = a4;
      v18 = a3 == 0 ? 4096 : 36864;
      if ( v20 )
      {
        if ( *(_BYTE *)(v20 + 80) == 2 )
        {
          v32 = *(__m128i **)(v20 + 88);
        }
        else
        {
          v29 = a4;
          v32 = (__m128i *)sub_724D50(6);
          sub_82D750(v30, 0, 0, v15, v32, a8);
          v18 = a3 == 0 ? 4096 : 36864;
          v16 = v29;
          if ( *a8 )
            return a1;
        }
        if ( v32 )
        {
          v21 = v32;
          goto LABEL_17;
        }
      }
    }
  }
  v21 = sub_743600(a2, v16, a5, v15, a6, v13 | (unsigned int)v18, a8, a9, a10);
  if ( *a8 )
    return a1;
  v22 = (__m128i *)v21;
  if ( !v21 )
    v22 = a10;
  v32 = v22;
LABEL_17:
  if ( (v13 & 4) != 0 )
    goto LABEL_26;
  v23 = v32[8].m128i_i64[0];
  if ( !sub_7306C0((__int64)v15) && !sub_7306C0(v23) )
    goto LABEL_26;
  v24 = v32[10].m128i_i8[13];
  if ( v24 == 10 )
  {
    v25 = sub_730B80((__int64)v32);
  }
  else
  {
    if ( (unsigned __int8)(v24 - 6) > 1u && !(unsigned int)sub_8D32B0(v32[8].m128i_i64[0]) )
      goto LABEL_26;
    v25 = sub_730990((__int64)v32);
  }
  if ( !v25 && v15 != (_QWORD *)v23 && !(unsigned int)sub_8DED30(v15, v23, 1) )
  {
LABEL_34:
    *a8 = 1;
    return a1;
  }
LABEL_26:
  if ( !(unsigned int)sub_728A90((__int64)v32, (__int64)v15, a3, (a7 & 0x80) != 0, &v34) )
    goto LABEL_34;
  v26 = *(_QWORD **)(a1 + 128);
  if ( v26 == v15 || v26 && v15 && dword_4F07588 && (v28 = v15[4], v26[4] == v28) && v28 )
  {
    if ( a2 == v21 )
      return a1;
  }
  if ( v21 )
  {
    *a10 = _mm_loadu_si128(v21);
    a10[1] = _mm_loadu_si128(v21 + 1);
    a10[2] = _mm_loadu_si128(v21 + 2);
    a10[3] = _mm_loadu_si128(v21 + 3);
    a10[4] = _mm_loadu_si128(v21 + 4);
    a10[5] = _mm_loadu_si128(v21 + 5);
    a10[6] = _mm_loadu_si128(v21 + 6);
    a10[7] = _mm_loadu_si128(v21 + 7);
    a10[8] = _mm_loadu_si128(v21 + 8);
    a10[9] = _mm_loadu_si128(v21 + 9);
    a10[10] = _mm_loadu_si128(v21 + 10);
    a10[11] = _mm_loadu_si128(v21 + 11);
    a10[12] = _mm_loadu_si128(v21 + 12);
  }
  sub_7115B0(a10, (__int64)v15, a3 == 0, 1, 1, 1, 0, 0, 1u, v34, 0, v35, (_DWORD *)v35 + 1, a6);
  result = 0;
  if ( v35[0] )
    goto LABEL_34;
  return result;
}
