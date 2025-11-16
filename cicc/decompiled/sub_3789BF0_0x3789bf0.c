// Function: sub_3789BF0
// Address: 0x3789bf0
//
__m128i *__fastcall sub_3789BF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  const __m128i *v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r13
  __int64 v8; // rdx
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int64 v12; // rdx
  __m128i v13; // xmm4
  __int64 v14; // rdx
  __m128i *v15; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i v21; // rax
  __int64 v22; // rax
  int v23; // edx
  int v24; // edx
  __int64 v25; // r9
  int v26; // edx
  _QWORD *v27; // [rsp+0h] [rbp-170h]
  char v28; // [rsp+8h] [rbp-168h]
  unsigned __int64 v29; // [rsp+10h] [rbp-160h]
  __int16 v30; // [rsp+18h] [rbp-158h]
  unsigned __int8 v31; // [rsp+1Fh] [rbp-151h]
  __int64 v32; // [rsp+60h] [rbp-110h] BYREF
  int v33; // [rsp+68h] [rbp-108h]
  __m128i v34; // [rsp+70h] [rbp-100h] BYREF
  __int64 v35[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int128 v36; // [rsp+90h] [rbp-E0h] BYREF
  __int128 v37; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v38; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i v39; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+D0h] [rbp-A0h]
  __int64 v41; // [rsp+D8h] [rbp-98h]
  __int64 v42; // [rsp+E0h] [rbp-90h]
  __int64 v43; // [rsp+E8h] [rbp-88h]
  __int64 v44; // [rsp+F0h] [rbp-80h]
  __int64 v45; // [rsp+F8h] [rbp-78h]
  _OWORD v46[2]; // [rsp+100h] [rbp-70h] BYREF
  __m128i v47; // [rsp+120h] [rbp-50h] BYREF
  __m128i v48; // [rsp+130h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a2 + 80);
  v32 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v32, v3, 1);
  v4 = *(_QWORD *)(a2 + 104);
  v33 = *(_DWORD *)(a2 + 72);
  v28 = *(_BYTE *)(a2 + 33);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = v5->m128i_i64[0];
  v7 = v5->m128i_u64[1];
  v35[1] = v4;
  v8 = *(_QWORD *)(a2 + 112);
  v9 = _mm_loadu_si128(v5 + 5);
  DWORD2(v36) = 0;
  v29 = v6;
  LOWORD(v6) = *(_WORD *)(a2 + 96);
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = _mm_loadu_si128((const __m128i *)(v8 + 56));
  v34 = v9;
  LOWORD(v35[0]) = v6;
  LOBYTE(v6) = *(_BYTE *)(v8 + 34);
  v46[0] = v10;
  v31 = v6;
  LOWORD(v6) = *(_WORD *)(v8 + 32);
  DWORD2(v37) = 0;
  v46[1] = v11;
  v12 = v5[3].m128i_i64[0];
  *(_QWORD *)&v36 = 0;
  *(_QWORD *)&v37 = 0;
  v30 = v6;
  sub_375E8D0((__int64)a1, v5[2].m128i_u64[1], v12, (__int64)&v36, (__int64)&v37);
  sub_33D0340((__int64)&v47, a1[1], v35);
  v13 = _mm_loadu_si128(&v48);
  v38 = _mm_loadu_si128(&v47);
  v39 = v13;
  if ( v47.m128i_i16[0] )
  {
    if ( v47.m128i_i16[0] == 1 || (unsigned __int16)(v47.m128i_i16[0] - 504) <= 7u )
      goto LABEL_26;
    v17 = *(_QWORD *)&byte_444C4A0[16 * v47.m128i_u16[0] - 16];
    if ( !v17 )
      goto LABEL_5;
  }
  else
  {
    v40 = sub_3007260((__int64)&v38);
    v41 = v14;
    if ( !v40 )
      goto LABEL_5;
    v17 = sub_3007260((__int64)&v38);
    v42 = v17;
    v43 = v18;
  }
  if ( (v17 & 7) != 0 )
    goto LABEL_5;
  if ( v39.m128i_i16[0] )
  {
    if ( v39.m128i_i16[0] != 1 && (unsigned __int16)(v39.m128i_i16[0] - 504) > 7u )
    {
      v19 = *(_QWORD *)&byte_444C4A0[16 * v39.m128i_u16[0] - 16];
      goto LABEL_13;
    }
LABEL_26:
    BUG();
  }
  v19 = sub_3007260((__int64)&v39);
  v44 = v19;
  v45 = v20;
LABEL_13:
  if ( v19 )
  {
    v27 = (_QWORD *)a1[1];
    v21.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v39);
    v47 = v21;
    if ( (v21.m128i_i8[0] & 7) == 0 )
    {
      v22 = *(_QWORD *)(a2 + 112);
      if ( (v28 & 4) != 0 )
      {
        *(_QWORD *)&v36 = sub_33F5040(
                            v27,
                            v29,
                            v7,
                            (__int64)&v32,
                            v36,
                            *((unsigned __int64 *)&v36 + 1),
                            v34.m128i_u64[0],
                            v34.m128i_u64[1],
                            *(_OWORD *)v22,
                            *(_QWORD *)(v22 + 16),
                            v38.m128i_i64[0],
                            v38.m128i_u64[1],
                            v31,
                            v30,
                            (__int64)v46);
        v47 = 0u;
        DWORD2(v36) = v23;
        v48.m128i_i32[0] = 0;
        v48.m128i_i8[4] = 0;
        sub_3777490((__int64)a1, a2, v38.m128i_u32[0], v38.m128i_i64[1], (__int64)&v47, (unsigned int *)&v34, v9, 0);
        *(_QWORD *)&v37 = sub_33F5040(
                            (_QWORD *)a1[1],
                            v29,
                            v7,
                            (__int64)&v32,
                            v37,
                            *((unsigned __int64 *)&v37 + 1),
                            v34.m128i_u64[0],
                            v34.m128i_u64[1],
                            *(_OWORD *)&v47,
                            v48.m128i_i64[0],
                            v39.m128i_i64[0],
                            v39.m128i_u64[1],
                            v31,
                            v30,
                            (__int64)v46);
      }
      else
      {
        *(_QWORD *)&v36 = sub_33F4560(
                            v27,
                            v29,
                            v7,
                            (__int64)&v32,
                            v36,
                            *((unsigned __int64 *)&v36 + 1),
                            v34.m128i_u64[0],
                            v34.m128i_u64[1],
                            *(_OWORD *)v22,
                            *(_QWORD *)(v22 + 16),
                            v31,
                            v30,
                            (__int64)v46);
        v47 = 0u;
        DWORD2(v36) = v26;
        v48.m128i_i32[0] = 0;
        v48.m128i_i8[4] = 0;
        sub_3777490((__int64)a1, a2, v38.m128i_u32[0], v38.m128i_i64[1], (__int64)&v47, (unsigned int *)&v34, v9, 0);
        *(_QWORD *)&v37 = sub_33F4560(
                            (_QWORD *)a1[1],
                            v29,
                            v7,
                            (__int64)&v32,
                            v37,
                            *((unsigned __int64 *)&v37 + 1),
                            v34.m128i_u64[0],
                            v34.m128i_u64[1],
                            *(_OWORD *)&v47,
                            v48.m128i_i64[0],
                            v31,
                            v30,
                            (__int64)v46);
      }
      DWORD2(v37) = v24;
      v15 = (__m128i *)sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v32, 1, 0, v25, v36, v37);
      goto LABEL_6;
    }
  }
LABEL_5:
  v15 = sub_3461110(v9, *a1, a2, a1[1]);
LABEL_6:
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  return v15;
}
