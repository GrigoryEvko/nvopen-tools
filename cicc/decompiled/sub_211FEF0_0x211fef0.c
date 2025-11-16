// Function: sub_211FEF0
// Address: 0x211fef0
//
__int64 __fastcall sub_211FEF0(__int64 *a1, unsigned __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r12
  __int64 v8; // rax
  char v9; // di
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // r13
  unsigned int v13; // esi
  bool v14; // cc
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __m128i v18; // xmm0
  int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r12
  int v23; // r11d
  unsigned __int8 v24; // dl
  unsigned __int16 v25; // di
  const __m128i *v26; // r9
  __int64 v27; // r12
  __int64 v28; // rdx
  __int64 v29; // r13
  __int8 v30; // r9
  __int64 v31; // rsi
  const void **v32; // r8
  __int64 *v33; // r14
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // r12
  unsigned int v38; // eax
  __int128 v39; // [rsp-10h] [rbp-B0h]
  int v40; // [rsp+0h] [rbp-A0h]
  __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  unsigned int v43; // [rsp+1Ch] [rbp-84h]
  __int8 v44; // [rsp+1Ch] [rbp-84h]
  __int64 v45; // [rsp+20h] [rbp-80h]
  const void **v46; // [rsp+20h] [rbp-80h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+30h] [rbp-70h] BYREF
  __int64 v49; // [rsp+38h] [rbp-68h]
  __int64 v50; // [rsp+40h] [rbp-60h] BYREF
  int v51; // [rsp+48h] [rbp-58h]
  __m128i v52; // [rsp+50h] [rbp-50h] BYREF
  const void **v53; // [rsp+60h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  LOBYTE(v48) = v9;
  v49 = v10;
  if ( v9 )
  {
    v11 = sub_211A7A0(v9);
    v12 = a1[1];
    v13 = v11;
    v14 = v11 <= 0x20;
    if ( v11 != 32 )
      goto LABEL_3;
LABEL_24:
    LOBYTE(v15) = 5;
    goto LABEL_6;
  }
  v38 = sub_1F58D40((__int64)&v48);
  v12 = a1[1];
  v13 = v38;
  v14 = v38 <= 0x20;
  if ( v38 == 32 )
    goto LABEL_24;
LABEL_3:
  if ( v14 )
  {
    if ( v13 == 8 )
    {
      LOBYTE(v15) = 3;
    }
    else
    {
      LOBYTE(v15) = 4;
      if ( v13 != 16 )
      {
        LOBYTE(v15) = 2;
        if ( v13 != 1 )
          goto LABEL_19;
      }
    }
LABEL_6:
    v16 = 0;
    goto LABEL_7;
  }
  if ( v13 == 64 )
  {
    LOBYTE(v15) = 6;
    goto LABEL_6;
  }
  if ( v13 == 128 )
  {
    LOBYTE(v15) = 7;
    goto LABEL_6;
  }
LABEL_19:
  v15 = sub_1F58CC0(*(_QWORD **)(v12 + 48), v13);
  v12 = a1[1];
  v5 = v15;
LABEL_7:
  v17 = *(_QWORD *)(a2 + 104);
  LOBYTE(v5) = v15;
  v47 = v16;
  v45 = v5;
  v18 = _mm_loadu_si128((const __m128i *)(v17 + 40));
  v52 = v18;
  v53 = *(const void ***)(v17 + 56);
  v43 = *(unsigned __int16 *)(v17 + 32);
  v19 = sub_1E34390(v17);
  v20 = *(_QWORD *)(a2 + 72);
  v21 = *(_QWORD *)(a2 + 104);
  v22 = *(_QWORD *)(a2 + 32);
  v23 = v19;
  v50 = v20;
  if ( v20 )
  {
    v41 = v21;
    v40 = v19;
    sub_1623A60((__int64)&v50, v20, 2);
    v23 = v40;
    v21 = v41;
  }
  v24 = *(_BYTE *)(a2 + 27);
  v25 = *(_WORD *)(a2 + 26);
  v51 = *(_DWORD *)(a2 + 64);
  v27 = sub_1D264C0(
          (_QWORD *)v12,
          (v25 >> 7) & 7,
          (v24 >> 2) & 3,
          v45,
          v47,
          (__int64)&v50,
          *(_OWORD *)v22,
          *(_QWORD *)(v22 + 40),
          *(_QWORD *)(v22 + 48),
          *(_OWORD *)(v22 + 80),
          *(_OWORD *)v21,
          *(_QWORD *)(v21 + 16),
          v45,
          v47,
          v23,
          v43,
          (__int64)&v52,
          0);
  v29 = v28;
  if ( v50 )
    sub_161E7C0((__int64)&v50, v50);
  sub_2013400((__int64)a1, a2, 1, v27, (__m128i *)1, v26);
  sub_1F40D10((__int64)&v52, *a1, *(_QWORD *)(a1[1] + 48), v48, v49);
  v30 = v52.m128i_i8[8];
  v31 = *(_QWORD *)(a2 + 72);
  v32 = v53;
  v33 = (__int64 *)a1[1];
  v52.m128i_i64[0] = v31;
  v34 = v52.m128i_u8[8];
  if ( v31 )
  {
    v46 = v53;
    v42 = v52.m128i_u8[8];
    v44 = v52.m128i_i8[8];
    sub_1623A60((__int64)&v52, v31, 2);
    v34 = v42;
    v32 = v46;
    v30 = v44;
  }
  v52.m128i_i32[2] = *(_DWORD *)(a2 + 64);
  if ( (_BYTE)v48 == 8 )
  {
    v35 = 160;
  }
  else
  {
    if ( v30 != 8 )
      sub_16BD130("Attempt at an invalid promotion-related conversion", 1u);
    v35 = 161;
  }
  *((_QWORD *)&v39 + 1) = v29;
  *(_QWORD *)&v39 = v27;
  v36 = sub_1D309E0(v33, v35, (__int64)&v52, v34, v32, 0, *(double *)v18.m128i_i64, a4, a5, v39);
  if ( v52.m128i_i64[0] )
    sub_161E7C0((__int64)&v52, v52.m128i_i64[0]);
  return v36;
}
