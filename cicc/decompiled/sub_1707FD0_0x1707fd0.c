// Function: sub_1707FD0
// Address: 0x1707fd0
//
_QWORD *__fastcall sub_1707FD0(const __m128i *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  int v4; // r13d
  _QWORD *result; // rax
  unsigned __int8 *v6; // r11
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  char v10; // al
  int v11; // eax
  unsigned int v12; // r13d
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  _QWORD *v15; // rax
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  _QWORD *v18; // r10
  __int64 v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  unsigned __int8 *v24; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v25; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v27; // [rsp+20h] [rbp-B0h]
  __int64 v28; // [rsp+28h] [rbp-A8h]
  int v29; // [rsp+34h] [rbp-9Ch]
  __int64 v30; // [rsp+38h] [rbp-98h]
  unsigned __int8 *v31; // [rsp+40h] [rbp-90h]
  __int64 v32; // [rsp+40h] [rbp-90h]
  __int64 v33; // [rsp+40h] [rbp-90h]
  bool v34; // [rsp+48h] [rbp-88h]
  _QWORD *v35; // [rsp+48h] [rbp-88h]
  __int64 v36[2]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v37; // [rsp+60h] [rbp-70h]
  __m128i v38; // [rsp+70h] [rbp-60h] BYREF
  __m128i v39; // [rsp+80h] [rbp-50h]
  unsigned __int8 *v40; // [rsp+90h] [rbp-40h]

  v4 = a2[16];
  if ( *(_BYTE *)(a3 + 16) != 79 )
    return 0;
  v26 = *(_QWORD *)(a3 - 72);
  if ( !v26 )
    return 0;
  v27 = *(unsigned __int8 **)(a3 - 48);
  if ( !v27 )
    return 0;
  v6 = *(unsigned __int8 **)(a3 - 24);
  if ( !v6 )
    return 0;
  if ( *(_BYTE *)(a4 + 16) != 79 )
    return 0;
  if ( v26 != *(_QWORD *)(a4 - 72) )
    return 0;
  v24 = *(unsigned __int8 **)(a4 - 48);
  if ( !v24 )
    return 0;
  v25 = *(unsigned __int8 **)(a4 - 24);
  if ( !v25 )
    return 0;
  v7 = *(_QWORD *)(a3 + 8);
  if ( !v7 )
    goto LABEL_31;
  v34 = 0;
  if ( *(_QWORD *)(v7 + 8) )
    goto LABEL_14;
  v8 = *(_QWORD *)(a4 + 8);
  if ( v8 )
    v34 = *(_QWORD *)(v8 + 8) == 0;
  else
LABEL_31:
    v34 = 0;
LABEL_14:
  v9 = a1->m128i_i64[1];
  v29 = *(_DWORD *)(v9 + 40);
  v30 = *(_QWORD *)(v9 + 32);
  v10 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  if ( (unsigned __int8)(v10 - 1) <= 5u || (_BYTE)v4 == 76 )
  {
    v31 = *(unsigned __int8 **)(a3 - 24);
    v11 = sub_15F24E0((__int64)a2);
    v6 = v31;
    *(_DWORD *)(v9 + 40) = v11;
  }
  v12 = v4 - 24;
  v40 = a2;
  v13 = _mm_loadu_si128(a1 + 167);
  v14 = _mm_loadu_si128(a1 + 168);
  v28 = (__int64)v6;
  v38 = v13;
  v39 = v14;
  v15 = sub_13E1140(v12, v6, v25, &v38);
  v40 = a2;
  v16 = _mm_loadu_si128(a1 + 167);
  v32 = (__int64)v15;
  v17 = _mm_loadu_si128(a1 + 168);
  v38 = v16;
  v39 = v17;
  v18 = sub_13E1140(v12, v27, v24, &v38);
  if ( v32 )
  {
    if ( v18 )
    {
      v19 = a1->m128i_i64[1];
      v39.m128i_i16[0] = 257;
      result = sub_1707C10(v19, v26, (__int64)v18, v32, v38.m128i_i64, 0);
      goto LABEL_22;
    }
    if ( v34 )
    {
      v20 = a1->m128i_i64[1];
      v37 = 257;
      v39.m128i_i16[0] = 257;
      v21 = sub_17066B0(
              v20,
              v12,
              (__int64)v27,
              (__int64)v24,
              v36,
              0,
              *(double *)v13.m128i_i64,
              *(double *)v14.m128i_i64,
              *(double *)v16.m128i_i64);
      result = sub_1707C10(v20, v26, v21, v32, v38.m128i_i64, 0);
      goto LABEL_22;
    }
  }
  else
  {
    v33 = (__int64)v18;
    if ( v18 && v34 )
    {
      v22 = a1->m128i_i64[1];
      v39.m128i_i16[0] = 257;
      v37 = 257;
      v23 = sub_17066B0(
              v22,
              v12,
              v28,
              (__int64)v25,
              v36,
              0,
              *(double *)v13.m128i_i64,
              *(double *)v14.m128i_i64,
              *(double *)v16.m128i_i64);
      result = sub_1707C10(v22, v26, v33, v23, v38.m128i_i64, 0);
LABEL_22:
      if ( result )
      {
        v35 = result;
        sub_164B7C0((__int64)result, (__int64)a2);
        result = v35;
      }
      goto LABEL_24;
    }
  }
  result = 0;
LABEL_24:
  *(_DWORD *)(v9 + 40) = v29;
  *(_QWORD *)(v9 + 32) = v30;
  return result;
}
