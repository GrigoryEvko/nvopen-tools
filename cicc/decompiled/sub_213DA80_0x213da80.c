// Function: sub_213DA80
// Address: 0x213da80
//
unsigned __int64 __fastcall sub_213DA80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // r9
  __int64 v11; // rsi
  __int128 *v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rax
  char v15; // di
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // di
  int v19; // edx
  int v20; // r8d
  __int64 *v21; // r15
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned __int8 v24; // al
  __int128 v25; // rax
  unsigned int v26; // edx
  unsigned __int64 result; // rax
  char v28; // r8
  unsigned int v29; // eax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // di
  __int64 v34; // rax
  int v35; // r13d
  int v36; // eax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  int v40; // esi
  __int64 v41; // r13
  unsigned int v42; // esi
  unsigned int v43; // eax
  unsigned __int64 v44; // rdx
  unsigned int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdi
  const void ***v49; // rdx
  unsigned int v50; // edx
  unsigned int v51; // eax
  __int128 v52; // [rsp-10h] [rbp-100h]
  __int128 *v53; // [rsp+8h] [rbp-E8h]
  unsigned int v54; // [rsp+10h] [rbp-E0h]
  __int8 v55; // [rsp+1Bh] [rbp-D5h]
  unsigned int v56; // [rsp+1Ch] [rbp-D4h]
  __int64 v57; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v58; // [rsp+28h] [rbp-C8h]
  __int64 v59; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v61; // [rsp+38h] [rbp-B8h]
  int v62; // [rsp+38h] [rbp-B8h]
  __m128i v63; // [rsp+70h] [rbp-80h] BYREF
  __int64 v64; // [rsp+80h] [rbp-70h] BYREF
  int v65; // [rsp+88h] [rbp-68h]
  char v66[8]; // [rsp+90h] [rbp-60h] BYREF
  __int64 v67; // [rsp+98h] [rbp-58h]
  __m128i v68; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v69; // [rsp+B0h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v68,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = a2;
  v63.m128i_i8[0] = v68.m128i_i8[8];
  v11 = *(_QWORD *)(a2 + 72);
  v64 = v11;
  v63.m128i_i64[1] = v69;
  if ( v11 )
  {
    sub_1623A60((__int64)&v64, v11, 2);
    v10 = a2;
  }
  v12 = *(__int128 **)(v10 + 32);
  v13 = _mm_loadu_si128(&v63);
  v65 = *(_DWORD *)(v10 + 64);
  v58 = *(_QWORD *)v12;
  v61 = *(_QWORD *)v12;
  v57 = *((_QWORD *)v12 + 1);
  v59 = 16LL * *((unsigned int *)v12 + 2);
  v14 = *(_QWORD *)(*(_QWORD *)v12 + 40LL) + v59;
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v68 = v13;
  v66[0] = v15;
  v67 = v16;
  if ( v15 == v63.m128i_i8[0] )
  {
    if ( v63.m128i_i8[0] || v16 == v68.m128i_i64[1] )
      goto LABEL_5;
LABEL_31:
    v55 = v63.m128i_i8[0];
    v53 = v12;
    v51 = sub_1F58D40((__int64)v66);
    v28 = v55;
    v56 = v51;
    if ( !v55 )
      goto LABEL_32;
LABEL_14:
    v29 = sub_2127930(v28);
    v12 = v53;
    goto LABEL_15;
  }
  if ( !v15 )
    goto LABEL_31;
  v53 = v12;
  v56 = sub_2127930(v15);
  if ( v28 )
    goto LABEL_14;
LABEL_32:
  v29 = sub_1F58D40((__int64)&v68);
  v12 = v53;
LABEL_15:
  if ( v29 < v56 )
  {
    v30 = sub_2138AD0(a1, v58, v57);
    sub_200E870(a1, v30, v31, a3, a4, v13, a6, a7);
    v32 = *(_QWORD *)(v61 + 40) + v59;
    v33 = *(_BYTE *)v32;
    v34 = *(_QWORD *)(v32 + 8);
    v68.m128i_i8[0] = v33;
    v68.m128i_i64[1] = v34;
    if ( v33 )
      v35 = sub_2127930(v33);
    else
      v35 = sub_1F58D40((__int64)&v68);
    if ( v63.m128i_i8[0] )
      v36 = sub_2127930(v63.m128i_i8[0]);
    else
      v36 = sub_1F58D40((__int64)&v63);
    v40 = v35;
    v41 = *(_QWORD *)(a1 + 8);
    v42 = v40 - v36;
    if ( v42 == 32 )
    {
      LOBYTE(v43) = 5;
    }
    else if ( v42 > 0x20 )
    {
      if ( v42 == 64 )
      {
        LOBYTE(v43) = 6;
      }
      else
      {
        if ( v42 != 128 )
        {
LABEL_34:
          v43 = sub_1F58CC0(*(_QWORD **)(v41 + 48), v42);
          v54 = v43;
          goto LABEL_25;
        }
        LOBYTE(v43) = 7;
      }
    }
    else if ( v42 == 8 )
    {
      LOBYTE(v43) = 3;
    }
    else
    {
      LOBYTE(v43) = 4;
      if ( v42 != 16 )
      {
        LOBYTE(v43) = 2;
        if ( v42 != 1 )
          goto LABEL_34;
      }
    }
    v44 = 0;
LABEL_25:
    v45 = v54;
    LOBYTE(v45) = v43;
    v46 = sub_1D2EF30((_QWORD *)v41, v45, v44, v37, v38, v39);
    v48 = v47;
    v49 = (const void ***)(*(_QWORD *)(*a4 + 40) + 16LL * *((unsigned int *)a4 + 2));
    *((_QWORD *)&v52 + 1) = v48;
    *(_QWORD *)&v52 = v46;
    *a4 = (__int64)sub_1D332F0(
                     (__int64 *)v41,
                     148,
                     (__int64)&v64,
                     *(unsigned __int8 *)v49,
                     v49[1],
                     0,
                     *(double *)v13.m128i_i64,
                     a6,
                     a7,
                     *a4,
                     a4[1],
                     v52);
    result = v50;
    *((_DWORD *)a4 + 2) = v50;
    goto LABEL_26;
  }
LABEL_5:
  v17 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          142,
          (__int64)&v64,
          v63.m128i_u32[0],
          (const void **)v63.m128i_i64[1],
          0,
          *(double *)v13.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *v12);
  v18 = v63.m128i_i8[0];
  *(_QWORD *)a3 = v17;
  *(_DWORD *)(a3 + 8) = v19;
  if ( v18 )
    v20 = sub_2127930(v18);
  else
    v20 = sub_1F58D40((__int64)&v63);
  v21 = *(__int64 **)(a1 + 8);
  v62 = v20;
  v22 = sub_1E0A0C0(v21[4]);
  v23 = 8 * sub_15A9520(v22, 0);
  if ( v23 == 32 )
  {
    v24 = 5;
  }
  else if ( v23 > 0x20 )
  {
    v24 = 6;
    if ( v23 != 64 )
    {
      v24 = 0;
      if ( v23 == 128 )
        v24 = 7;
    }
  }
  else
  {
    v24 = 3;
    if ( v23 != 8 )
      v24 = 4 * (v23 == 16);
  }
  *(_QWORD *)&v25 = sub_1D38BB0((__int64)v21, (unsigned int)(v62 - 1), (__int64)&v64, v24, 0, 0, v13, a6, a7, 0);
  *a4 = (__int64)sub_1D332F0(
                   v21,
                   123,
                   (__int64)&v64,
                   v63.m128i_u32[0],
                   (const void **)v63.m128i_i64[1],
                   0,
                   *(double *)v13.m128i_i64,
                   a6,
                   a7,
                   *(_QWORD *)a3,
                   *(_QWORD *)(a3 + 8),
                   v25);
  result = v26;
  *((_DWORD *)a4 + 2) = v26;
LABEL_26:
  if ( v64 )
    return sub_161E7C0((__int64)&v64, v64);
  return result;
}
