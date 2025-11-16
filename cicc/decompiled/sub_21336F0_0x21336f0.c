// Function: sub_21336F0
// Address: 0x21336f0
//
unsigned __int64 __fastcall sub_21336F0(
        __m128i **a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        double a5,
        double a6,
        __m128i a7)
{
  char *v9; // rax
  __int64 v10; // rsi
  unsigned __int8 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r13d
  __m128i *v14; // rsi
  __int64 v15; // rax
  __int64 *v16; // r15
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  int v19; // ecx
  unsigned __int64 result; // rax
  __int64 v21; // rsi
  const void ***v22; // rax
  int v23; // edx
  __int64 v24; // r9
  __int64 *v25; // rax
  __int128 v26; // [rsp-10h] [rbp-B0h]
  unsigned __int64 v27; // [rsp-10h] [rbp-B0h]
  __int64 v28; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v29; // [rsp+17h] [rbp-89h]
  __int64 v31; // [rsp+20h] [rbp-80h] BYREF
  int v32; // [rsp+28h] [rbp-78h]
  _OWORD v33[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v34[10]; // [rsp+50h] [rbp-50h] BYREF

  v9 = *(char **)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v31 = v10;
  v13 = v11;
  if ( v10 )
  {
    v29 = v11;
    v28 = v12;
    sub_1623A60((__int64)&v31, v10, 2);
    v11 = v29;
    v12 = v28;
  }
  v14 = *a1;
  v32 = *(_DWORD *)(a2 + 64);
  v15 = *(_QWORD *)(a2 + 32);
  v16 = (__int64 *)a1[1];
  v17 = _mm_loadu_si128((const __m128i *)v15);
  v18 = _mm_loadu_si128((const __m128i *)(v15 + 40));
  v33[0] = v17;
  v33[1] = v18;
  if ( !v11 )
  {
    v19 = 462;
    goto LABEL_6;
  }
  if ( v14[155].m128i_i8[259 * v11 + 4] != 4 )
  {
    v19 = 26;
    if ( v11 != 4 )
    {
      v19 = 27;
      if ( v11 != 5 )
      {
        v19 = 28;
        if ( v11 != 6 )
        {
          v19 = 29;
          if ( v11 != 7 )
            v19 = 462;
        }
      }
    }
LABEL_6:
    sub_20BE530((__int64)v34, v14, (__int64)v16, v19, v13, v12, v17, v18, a7, (__int64)v33, 2u, 0, (__int64)&v31, 0, 1);
    result = sub_200E870((__int64)a1, v34[0], v34[1], a3, a4, v17, *(double *)v18.m128i_i64, a7);
    v21 = v31;
    if ( !v31 )
      return result;
    return sub_161E7C0((__int64)&v31, v21);
  }
  v22 = (const void ***)sub_1D252B0((__int64)v16, v13, v12, v13, v12);
  *((_QWORD *)&v26 + 1) = 2;
  *(_QWORD *)&v26 = v33;
  v25 = sub_1D36D80(v16, 62, (__int64)&v31, v22, v23, *(double *)v17.m128i_i64, *(double *)v18.m128i_i64, a7, v24, v26);
  sub_200E870((__int64)a1, (__int64)v25, 0, a3, a4, v17, *(double *)v18.m128i_i64, a7);
  v21 = v31;
  result = v27;
  if ( !v31 )
    return result;
  return sub_161E7C0((__int64)&v31, v21);
}
