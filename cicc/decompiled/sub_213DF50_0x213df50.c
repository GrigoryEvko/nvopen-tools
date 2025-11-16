// Function: sub_213DF50
// Address: 0x213df50
//
unsigned __int64 __fastcall sub_213DF50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int128 *v11; // rdx
  __m128i v12; // xmm0
  __int64 v13; // r15
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // rax
  __int64 v17; // rax
  const void **v18; // r8
  int v19; // edx
  int v20; // edx
  unsigned __int64 result; // rax
  char v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  char v27; // di
  __int64 v28; // rax
  int v29; // r14d
  int v30; // eax
  int v31; // esi
  __int64 v32; // r14
  unsigned int v33; // esi
  unsigned int v34; // eax
  __int64 v35; // r9
  unsigned int v36; // ebx
  unsigned int v37; // edx
  unsigned int v38; // eax
  __int64 v39; // rdx
  unsigned __int64 v40; // [rsp+0h] [rbp-100h]
  __int128 *v41; // [rsp+18h] [rbp-E8h]
  unsigned int v42; // [rsp+20h] [rbp-E0h]
  __int8 v43; // [rsp+2Bh] [rbp-D5h]
  unsigned int v44; // [rsp+2Ch] [rbp-D4h]
  __int64 v45; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v46; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-C0h]
  __m128i v49; // [rsp+80h] [rbp-80h] BYREF
  __int64 v50; // [rsp+90h] [rbp-70h] BYREF
  int v51; // [rsp+98h] [rbp-68h]
  char v52[8]; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v53; // [rsp+A8h] [rbp-58h]
  __m128i v54; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v55; // [rsp+C0h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v54,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = *(_QWORD *)(a2 + 72);
  v49.m128i_i8[0] = v54.m128i_i8[8];
  v50 = v10;
  v49.m128i_i64[1] = v55;
  if ( v10 )
    sub_1623A60((__int64)&v50, v10, 2);
  v11 = *(__int128 **)(a2 + 32);
  v12 = _mm_loadu_si128(&v49);
  v51 = *(_DWORD *)(a2 + 64);
  v46 = *(_QWORD *)v11;
  v47 = *(_QWORD *)v11;
  v45 = *((_QWORD *)v11 + 1);
  v13 = 16LL * *((unsigned int *)v11 + 2);
  v14 = *(_QWORD *)(*(_QWORD *)v11 + 40LL) + v13;
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v54 = v12;
  v52[0] = v15;
  v53 = v16;
  if ( v15 == v49.m128i_i8[0] )
  {
    if ( v49.m128i_i8[0] || v16 == v54.m128i_i64[1] )
      goto LABEL_5;
LABEL_24:
    v43 = v49.m128i_i8[0];
    v41 = v11;
    v38 = sub_1F58D40((__int64)v52);
    v22 = v43;
    v44 = v38;
    if ( !v43 )
      goto LABEL_25;
LABEL_11:
    v23 = sub_2127930(v22);
    v11 = v41;
    goto LABEL_12;
  }
  if ( !v15 )
    goto LABEL_24;
  v41 = v11;
  v44 = sub_2127930(v15);
  if ( v22 )
    goto LABEL_11;
LABEL_25:
  v23 = sub_1F58D40((__int64)&v54);
  v11 = v41;
LABEL_12:
  if ( v23 >= v44 )
  {
LABEL_5:
    v17 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            143,
            (__int64)&v50,
            v49.m128i_u32[0],
            (const void **)v49.m128i_i64[1],
            0,
            *(double *)v12.m128i_i64,
            a6,
            *(double *)a7.m128i_i64,
            *v11);
    v18 = (const void **)v49.m128i_i64[1];
    *(_QWORD *)a3 = v17;
    *(_DWORD *)(a3 + 8) = v19;
    *a4 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v50, v49.m128i_u32[0], v18, 0, v12, a6, a7, 0);
    *((_DWORD *)a4 + 2) = v20;
    result = v40;
    goto LABEL_6;
  }
  v24 = sub_2138AD0(a1, v46, v45);
  sub_200E870(a1, v24, v25, a3, a4, v12, a6, a7);
  v26 = v13 + *(_QWORD *)(v47 + 40);
  v27 = *(_BYTE *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  v54.m128i_i8[0] = v27;
  v54.m128i_i64[1] = v28;
  if ( v27 )
    v29 = sub_2127930(v27);
  else
    v29 = sub_1F58D40((__int64)&v54);
  if ( v49.m128i_i8[0] )
    v30 = sub_2127930(v49.m128i_i8[0]);
  else
    v30 = sub_1F58D40((__int64)&v49);
  v31 = v29;
  v32 = *(_QWORD *)(a1 + 8);
  v33 = v31 - v30;
  if ( v33 == 32 )
  {
    LOBYTE(v34) = 5;
  }
  else if ( v33 > 0x20 )
  {
    if ( v33 == 64 )
    {
      LOBYTE(v34) = 6;
    }
    else
    {
      if ( v33 != 128 )
      {
LABEL_27:
        v34 = sub_1F58CC0(*(_QWORD **)(v32 + 48), v33);
        v42 = v34;
        v35 = v39;
        goto LABEL_22;
      }
      LOBYTE(v34) = 7;
    }
  }
  else if ( v33 == 8 )
  {
    LOBYTE(v34) = 3;
  }
  else
  {
    LOBYTE(v34) = 4;
    if ( v33 != 16 )
    {
      LOBYTE(v34) = 2;
      if ( v33 != 1 )
        goto LABEL_27;
    }
  }
  v35 = 0;
LABEL_22:
  v36 = v42;
  LOBYTE(v36) = v34;
  *a4 = sub_1D3BC50((__int64 *)v32, *a4, a4[1], (__int64)&v50, v36, v35, v12, a6, a7);
  result = v37;
  *((_DWORD *)a4 + 2) = v37;
LABEL_6:
  if ( v50 )
    return sub_161E7C0((__int64)&v50, v50);
  return result;
}
