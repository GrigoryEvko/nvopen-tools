// Function: sub_10A6920
// Address: 0x10a6920
//
unsigned __int8 *__fastcall sub_10A6920(const __m128i *a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  _BYTE *v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v13; // rbx
  bool v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 *v25; // rdi
  unsigned int v26; // r10d
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // rax
  __int64 v31; // r14
  unsigned __int64 v32; // r10
  __int64 v33; // rcx
  __int64 v34; // r8
  char v35; // al
  __int64 v36; // r10
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 *v39; // r13
  __m128i v40; // rax
  __int64 v41; // rax
  unsigned __int8 *v42; // rax
  char v43; // zf
  char v44; // al
  __m128i v45; // rax
  __int64 v46; // rax
  __int64 *v47; // r13
  __m128i v48; // rax
  __int64 v49; // rax
  char v50; // al
  __int64 *v51; // r13
  __m128i v52; // rax
  __int64 v53; // rax
  unsigned __int8 *v54; // rax
  __int64 v55; // [rsp+0h] [rbp-D0h]
  __int64 v56; // [rsp+10h] [rbp-C0h]
  _BYTE *v57; // [rsp+10h] [rbp-C0h]
  __int64 v58; // [rsp+10h] [rbp-C0h]
  __int64 *v59; // [rsp+10h] [rbp-C0h]
  __int64 v60; // [rsp+10h] [rbp-C0h]
  unsigned int v61; // [rsp+18h] [rbp-B8h]
  _BYTE *v62; // [rsp+18h] [rbp-B8h]
  _BYTE *v63; // [rsp+28h] [rbp-A8h] BYREF
  _BYTE *v64; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v65; // [rsp+38h] [rbp-98h] BYREF
  char v66; // [rsp+3Ch] [rbp-94h]
  __int64 v67[2]; // [rsp+40h] [rbp-90h] BYREF
  __m128i v68; // [rsp+50h] [rbp-80h] BYREF
  __m128i v69; // [rsp+60h] [rbp-70h]
  unsigned __int64 v70; // [rsp+70h] [rbp-60h]
  __int64 v71; // [rsp+78h] [rbp-58h]
  __m128i v72; // [rsp+80h] [rbp-50h]
  __int64 v73; // [rsp+90h] [rbp-40h]

  v2 = (unsigned __int8 *)a2;
  v5 = *(_BYTE **)(a2 - 32);
  v6 = a1[10].m128i_i64[0];
  v68 = _mm_loadu_si128(a1 + 6);
  v70 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v63 = v5;
  v71 = a2;
  v73 = v6;
  v69 = _mm_loadu_si128(a1 + 7);
  v72 = _mm_loadu_si128(a1 + 9);
  v7 = sub_B45210(a2);
  v10 = sub_1008810(v5, v7, (__int64)&v68, v8, v9);
  if ( !v10 )
  {
    v2 = sub_10A6470((unsigned __int8 *)a2);
    if ( v2 )
      return v2;
    v13 = (__int64)v63;
    v14 = sub_B451E0(a2);
    v15 = *((_QWORD *)v63 + 2);
    if ( v14 )
    {
      if ( !v15 )
        return v2;
      if ( *(_QWORD *)(v15 + 8) )
        return 0;
      if ( *v63 == 45 )
      {
        v37 = *((_QWORD *)v63 - 8);
        if ( v37 )
        {
          v38 = *((_QWORD *)v63 - 4);
          if ( v38 )
          {
            LOWORD(v70) = 257;
            v2 = (unsigned __int8 *)sub_B504D0(16, v38, v37, (__int64)&v68, 0, 0);
            sub_B45260(v2, a2, 1);
            return v2;
          }
        }
      }
    }
    else if ( !v15 || *(_QWORD *)(v15 + 8) )
    {
      return 0;
    }
    v16 = sub_10A1E40((__int64)a1, (__int64)v63, a2);
    if ( v16 )
      return sub_F162A0((__int64)a1, a2, v16);
    v19 = *v63;
    if ( *v63 != 86 )
    {
LABEL_25:
      if ( v19 != 85 )
        return 0;
      v20 = *(_QWORD *)(v13 - 32);
      if ( !v20 )
        return 0;
      if ( *(_BYTE *)v20 )
        return 0;
      if ( *(_QWORD *)(v20 + 24) != *(_QWORD *)(v13 + 80) )
        return 0;
      if ( *(_DWORD *)(v20 + 36) != 26 )
        return 0;
      v21 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
      v22 = *(_QWORD *)(v13 - 32 * v21);
      if ( !v22 )
        return 0;
      v23 = 32 * (1 - v21);
      if ( !*(_QWORD *)(v13 + v23) )
        return 0;
      v56 = *(_QWORD *)(v13 + v23);
      v24 = sub_B45210(a2);
      v25 = (__int64 *)a1[2].m128i_i64[0];
      v26 = v24;
      v27 = *(_BYTE *)(v13 + 1) >> 1;
      if ( v27 != 127 )
        v26 &= v27;
      LOWORD(v70) = 257;
      v61 = v26;
      v28 = sub_109D090(v25, v56, v26, 1, (__int64)&v68, 0);
      v29 = a1[2].m128i_i64[0];
      BYTE4(v67[0]) = 1;
      LODWORD(v67[0]) = v61;
      LOWORD(v70) = 257;
      v16 = sub_B33C40(v29, 0x1Au, v22, v28, v67[0], (__int64)&v68);
      return sub_F162A0((__int64)a1, a2, v16);
    }
    if ( (v63[7] & 0x40) != 0 )
    {
      v30 = (__int64 *)*((_QWORD *)v63 - 1);
      v31 = *v30;
      if ( !*v30 )
        return 0;
    }
    else
    {
      v30 = (__int64 *)&v63[-32 * (*((_DWORD *)v63 + 1) & 0x7FFFFFF)];
      v31 = *v30;
      if ( !*v30 )
        return 0;
    }
    v32 = v30[4];
    if ( !v32 )
      return 0;
    v62 = (_BYTE *)v30[8];
    if ( !v62 )
      return 0;
    v68.m128i_i64[0] = (__int64)&v64;
    v57 = (_BYTE *)v32;
    v67[0] = a2;
    v67[1] = (__int64)&v63;
    if ( (unsigned __int8)sub_995E90(&v68, v32, (__int64)&v64, v17, v18) )
    {
      v39 = (__int64 *)a1[2].m128i_i64[0];
      v40.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v62);
      LOWORD(v70) = 773;
      v68 = v40;
      v69.m128i_i64[0] = (__int64)".neg";
      sub_10A0170((__int64)&v65, a2);
      v41 = sub_109D090(v39, (__int64)v62, v65, v66, (__int64)&v68, 0);
      LOWORD(v70) = 257;
      v42 = sub_109FEA0(v31, (__int64)v64, v41, (const char **)&v68, 0, 0, 0);
      v43 = v64 == v62;
      v2 = v42;
    }
    else
    {
      v68.m128i_i64[0] = (__int64)&v64;
      v35 = sub_995E90(&v68, (unsigned __int64)v62, (__int64)&v64, v33, v34);
      v36 = (__int64)v57;
      if ( !v35 )
      {
        if ( *v57 <= 0x15u && (v50 = sub_109CE80(v57), v36 = (__int64)v57, v50)
          || *v62 <= 0x15u && (v58 = v36, v44 = sub_109CE80(v62), v36 = v58, v44) )
        {
          v55 = v36;
          v59 = (__int64 *)a1[2].m128i_i64[0];
          v45.m128i_i64[0] = (__int64)sub_BD5D20(v36);
          v69.m128i_i64[0] = (__int64)".neg";
          v68 = v45;
          LOWORD(v70) = 773;
          sub_10A0170((__int64)&v65, a2);
          v46 = sub_109D090(v59, v55, v65, v66, (__int64)&v68, 0);
          v47 = (__int64 *)a1[2].m128i_i64[0];
          v60 = v46;
          v48.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v62);
          v69.m128i_i64[0] = (__int64)".neg";
          v68 = v48;
          LOWORD(v70) = 773;
          sub_10A0170((__int64)&v65, a2);
          v49 = sub_109D090(v47, (__int64)v62, v65, v66, (__int64)&v68, 0);
          LOWORD(v70) = 257;
          v2 = sub_109FEA0(v31, v60, v49, (const char **)&v68, 0, 0, 0);
          sub_109CD10(v67, (__int64)v2, 1);
          return v2;
        }
        v19 = *(_BYTE *)v13;
        goto LABEL_25;
      }
      v51 = (__int64 *)a1[2].m128i_i64[0];
      v52.m128i_i64[0] = (__int64)sub_BD5D20((__int64)v57);
      LOWORD(v70) = 773;
      v68 = v52;
      v69.m128i_i64[0] = (__int64)".neg";
      sub_10A0170((__int64)&v65, a2);
      v53 = sub_109D090(v51, (__int64)v57, v65, v66, (__int64)&v68, 0);
      LOWORD(v70) = 257;
      v54 = sub_109FEA0(v31, v53, (__int64)v64, (const char **)&v68, 0, 0, 0);
      v43 = v64 == v57;
      v2 = v54;
    }
    sub_109CD10(v67, (__int64)v2, v43);
    return v2;
  }
  if ( !*(_QWORD *)(a2 + 16) )
    return 0;
  v11 = v10;
  sub_10A5FE0(a1[2].m128i_i64[1], a2);
  if ( a2 == v11 )
  {
    v11 = sub_ACADE0(*(__int64 ***)(a2 + 8));
    if ( *(_QWORD *)(v11 + 16) )
      goto LABEL_5;
LABEL_11:
    if ( *(_BYTE *)v11 > 0x1Cu && (*(_BYTE *)(v11 + 7) & 0x10) == 0 && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
      sub_BD6B90((unsigned __int8 *)v11, (unsigned __int8 *)a2);
    goto LABEL_5;
  }
  if ( !*(_QWORD *)(v11 + 16) )
    goto LABEL_11;
LABEL_5:
  sub_BD84D0(a2, v11);
  return v2;
}
