// Function: sub_1F7FF10
// Address: 0x1f7ff10
//
__int64 *__fastcall sub_1F7FF10(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9)
{
  __int64 v10; // r14
  __int64 v13; // rbx
  __int16 v14; // ax
  __int64 v16; // rax
  __int64 *v17; // rax
  __int16 v18; // di
  const __m128i *v19; // rax
  __int64 v20; // rdx
  __m128i v21; // xmm0
  unsigned __int8 *v22; // rsi
  int v23; // ecx
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // r9
  __int64 v27; // r14
  __int64 v28; // rdi
  __int64 v29; // r14
  int v30; // ecx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  int v36; // ecx
  int v37; // r8d
  int v38; // r9d
  __int64 v39; // rax
  unsigned __int8 v40; // r8
  int v41; // eax
  bool v42; // al
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r9
  __int64 v46; // r14
  unsigned __int8 v47; // r8
  unsigned int v48; // eax
  unsigned __int64 v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rdx
  __int64 *v52; // rax
  unsigned int v53; // eax
  const void **v54; // r8
  unsigned int v55; // ecx
  unsigned __int8 *v56; // rax
  const void **v57; // r12
  unsigned int v58; // ebx
  __int128 v59; // rax
  unsigned __int8 v60; // [rsp+8h] [rbp-E8h]
  unsigned __int8 v61; // [rsp+10h] [rbp-E0h]
  int v62; // [rsp+10h] [rbp-E0h]
  unsigned __int8 v63; // [rsp+10h] [rbp-E0h]
  unsigned int v64; // [rsp+10h] [rbp-E0h]
  unsigned __int8 v65; // [rsp+10h] [rbp-E0h]
  char v66; // [rsp+10h] [rbp-E0h]
  __int64 v67; // [rsp+18h] [rbp-D8h]
  __int64 v68; // [rsp+20h] [rbp-D0h]
  __int64 v69; // [rsp+28h] [rbp-C8h]
  __int64 v70; // [rsp+40h] [rbp-B0h]
  unsigned __int8 v71; // [rsp+40h] [rbp-B0h]
  __int64 v72; // [rsp+48h] [rbp-A8h]
  unsigned int v73; // [rsp+48h] [rbp-A8h]
  __int64 *v74; // [rsp+48h] [rbp-A8h]
  __int64 *v75; // [rsp+48h] [rbp-A8h]
  char v76[8]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v77; // [rsp+58h] [rbp-98h]
  __int64 *v78; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v79; // [rsp+68h] [rbp-88h]
  __int64 v80; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v81; // [rsp+78h] [rbp-78h]
  __int64 v82[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v83[2]; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v84; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v85; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v86; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v87; // [rsp+B8h] [rbp-38h]

  v10 = a4;
  v13 = a3;
  if ( *(_WORD *)(a3 + 24) == 118 && (v72 = a5, sub_1D23600((__int64)a1, *(_QWORD *)(*(_QWORD *)(a3 + 32) + 40LL))) )
  {
    v16 = *(_QWORD *)(v13 + 32);
    LODWORD(a5) = v72;
    *(_QWORD *)v72 = *(_QWORD *)(v16 + 40);
    *(_DWORD *)(v72 + 8) = *(_DWORD *)(v16 + 48);
    v17 = *(__int64 **)(v13 + 32);
    v13 = *v17;
    v10 = *((unsigned int *)v17 + 2);
    v14 = *(_WORD *)(a2 + 24);
    if ( v14 == 124 )
    {
LABEL_7:
      v18 = *(_WORD *)(v13 + 24);
      LOBYTE(a5) = v18 == 54;
      if ( v18 != 122 && v18 != 54 )
        return 0;
      v73 = 122;
      goto LABEL_10;
    }
  }
  else
  {
    v14 = *(_WORD *)(a2 + 24);
    if ( v14 == 124 )
      goto LABEL_7;
  }
  if ( v14 != 122 )
    return 0;
  v18 = *(_WORD *)(v13 + 24);
  if ( v18 != 124 && v18 != 56 )
    return 0;
  v73 = 124;
  LOBYTE(a5) = v18 == 56;
LABEL_10:
  v19 = *(const __m128i **)(a2 + 32);
  v20 = v19->m128i_i64[0];
  v21 = _mm_loadu_si128(v19);
  v22 = (unsigned __int8 *)(*(_QWORD *)(v19->m128i_i64[0] + 40) + 16LL * v19->m128i_u32[2]);
  v23 = *v22;
  v24 = *((_QWORD *)v22 + 1);
  v76[0] = v23;
  v77 = v24;
  if ( *(_WORD *)(v20 + 24) != v18 )
    return 0;
  v25 = *(_QWORD *)(v13 + 32);
  v26 = *(_QWORD *)(v20 + 32);
  if ( *(_QWORD *)v26 != *(_QWORD *)v25 )
    return 0;
  if ( *(_DWORD *)(v26 + 8) != *(_DWORD *)(v25 + 8) )
    return 0;
  v27 = 16 * v10;
  v28 = *(_QWORD *)(v13 + 40) + v27;
  v69 = v27;
  if ( (_BYTE)v23 != *(_BYTE *)v28 || *(_QWORD *)(v28 + 8) != v24 && !(_BYTE)v23 )
    return 0;
  v61 = a5;
  v70 = v20;
  v29 = sub_1D1ADA0(v19[2].m128i_i64[1], v19[3].m128i_i64[0], v20, v23, a5, v26);
  v68 = sub_1D1ADA0(
          *(_QWORD *)(*(_QWORD *)(v70 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(v70 + 32) + 48LL),
          v70,
          v30,
          v31,
          v32);
  v33 = *(_QWORD *)(v13 + 32);
  v34 = *(_QWORD *)(v33 + 48);
  v67 = sub_1D1ADA0(*(_QWORD *)(v33 + 40), v34, v35, v36, v37, v38);
  if ( !v29 )
    return 0;
  v39 = *(_QWORD *)(v29 + 88);
  v40 = v61;
  if ( *(_DWORD *)(v39 + 32) <= 0x40u )
  {
    if ( *(_QWORD *)(v39 + 24) )
      goto LABEL_19;
    return 0;
  }
  v62 = *(_DWORD *)(v39 + 32);
  v71 = v40;
  v41 = sub_16A57B0(v39 + 24);
  v40 = v71;
  if ( v62 == v41 )
    return 0;
LABEL_19:
  v63 = v40;
  if ( !v68 )
    return 0;
  v42 = sub_13D01C0(*(_QWORD *)(v68 + 88) + 24LL);
  if ( !v67 || v42 || sub_13D01C0(*(_QWORD *)(v67 + 88) + 24LL) )
    return 0;
  v60 = v63;
  v64 = sub_1D159C0((__int64)v76, v34, v43, v44, v63, v45);
  sub_13A38D0((__int64)&v86, *(_QWORD *)(v29 + 88) + 24LL);
  v46 = v64;
  v47 = v60;
  if ( v87 > 0x40 )
  {
    sub_16A8F40((__int64 *)&v86);
    v47 = v60;
  }
  else
  {
    v86 = ~v86 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v87);
  }
  v65 = v47;
  sub_16A7400((__int64)&v86);
  sub_16A7490((__int64)&v86, v46);
  v48 = v87;
  v87 = 0;
  v79 = v48;
  v78 = (__int64 *)v86;
  sub_135E100((__int64 *)&v86);
  if ( v79 > 0x40 )
    v49 = v78[(v79 - 1) >> 6];
  else
    v49 = (unsigned __int64)v78;
  if ( (v49 & (1LL << ((unsigned __int8)v79 - 1))) == 0 )
  {
    sub_13A38D0((__int64)&v80, *(_QWORD *)(v67 + 88) + 24LL);
    sub_13A38D0((__int64)v82, *(_QWORD *)(v68 + 88) + 24LL);
    sub_1F6DAA0((__int64)&v80, (__int64)v82, 0);
    if ( v65 )
    {
      LODWORD(v51) = (_DWORD)v78;
      if ( v79 > 0x40 )
        v51 = *v78;
      sub_1455760((__int64)v83, v81, v51);
      v85 = 1;
      v84 = 0;
      v87 = 1;
      v86 = 0;
      sub_16ADD10((__int64)&v80, (__int64)v83, &v84, &v86);
      if ( !sub_13A38F0((__int64)&v86, 0) || !sub_1455820((__int64)&v84, v82) )
      {
        sub_135E100((__int64 *)&v86);
        sub_135E100((__int64 *)&v84);
        sub_135E100(v83);
        v52 = 0;
LABEL_41:
        v75 = v52;
        sub_135E100(v82);
        sub_135E100(&v80);
        v50 = v75;
        goto LABEL_29;
      }
      sub_135E100((__int64 *)&v86);
      sub_135E100((__int64 *)&v84);
      sub_135E100(v83);
    }
    else
    {
      sub_16A5D10((__int64)&v84, (__int64)&v78, v81);
      if ( v85 > 0x40 )
        sub_16A8F40((__int64 *)&v84);
      else
        v84 = ~v84 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v85);
      sub_16A7400((__int64)&v84);
      sub_16A7200((__int64)&v84, &v80);
      v53 = v85;
      v85 = 0;
      v87 = v53;
      v86 = v84;
      v66 = sub_1455820((__int64)v82, &v86);
      sub_135E100((__int64 *)&v86);
      sub_135E100((__int64 *)&v84);
      if ( !v66 )
      {
        v52 = 0;
        goto LABEL_41;
      }
    }
    v54 = *(const void ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL)
                          + 8);
    v55 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
    v56 = (unsigned __int8 *)(*(_QWORD *)(v13 + 40) + v69);
    v57 = (const void **)*((_QWORD *)v56 + 1);
    v58 = *v56;
    *(_QWORD *)&v59 = sub_1D38970((__int64)a1, (__int64)&v78, a6, v55, v54, 0, v21, a8, a9, 0);
    v52 = sub_1D332F0(
            a1,
            v73,
            a6,
            v58,
            v57,
            0,
            *(double *)v21.m128i_i64,
            a8,
            a9,
            v21.m128i_i64[0],
            v21.m128i_u64[1],
            v59);
    goto LABEL_41;
  }
  v50 = 0;
LABEL_29:
  v74 = v50;
  sub_135E100((__int64 *)&v78);
  return v74;
}
