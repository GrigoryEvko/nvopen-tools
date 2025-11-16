// Function: sub_10B79B0
// Address: 0x10b79b0
//
unsigned __int8 *__fastcall sub_10B79B0(__m128i *a1, __int64 a2)
{
  unsigned __int8 *v2; // r15
  __int64 v3; // rax
  char v4; // al
  unsigned __int8 *v5; // rax
  __int64 v6; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r9
  __int64 v13; // rdx
  _QWORD *v14; // rcx
  __int64 v15; // r8
  bool v16; // zf
  _QWORD *v17; // rcx
  __int64 v18; // r8
  unsigned int **v19; // r13
  __int64 v20; // rax
  __int64 v21; // r9
  char v22; // al
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int64 v26; // rsi
  unsigned __int8 *v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // r8
  unsigned int **v30; // r13
  __int64 v31; // rax
  unsigned __int8 *v32; // r13
  unsigned __int8 *v33; // rdx
  __int64 v34; // r9
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  void *v38; // rax
  __int64 v39; // rbx
  __int64 *v40; // rsi
  __int64 v41; // rbx
  unsigned __int8 *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r13
  unsigned __int8 *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r9
  __int64 v49; // r9
  _BYTE *v50; // rdx
  __int64 v51; // [rsp+0h] [rbp-120h]
  void *v52; // [rsp+18h] [rbp-108h]
  _BYTE *v53; // [rsp+20h] [rbp-100h] BYREF
  _BYTE *v54; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v55; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v56; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v57; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+48h] [rbp-D8h] BYREF
  _BYTE **v59; // [rsp+50h] [rbp-D0h] BYREF
  _BYTE **v60; // [rsp+58h] [rbp-C8h]
  __int64 *v61; // [rsp+60h] [rbp-C0h]
  __m128i v62; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v63; // [rsp+80h] [rbp-A0h]
  _BYTE **v64; // [rsp+90h] [rbp-90h]
  _BYTE **v65; // [rsp+98h] [rbp-88h]
  __m128i v66; // [rsp+A0h] [rbp-80h]
  __int64 v67; // [rsp+B0h] [rbp-70h]
  _BYTE **v68; // [rsp+B8h] [rbp-68h]
  int v69; // [rsp+C0h] [rbp-60h]
  int v70; // [rsp+C8h] [rbp-58h]
  _BYTE **v71; // [rsp+D0h] [rbp-50h]
  int v72; // [rsp+D8h] [rbp-48h]
  _BYTE **v73; // [rsp+E0h] [rbp-40h]

  v2 = (unsigned __int8 *)a2;
  v3 = a1[10].m128i_i64[0];
  v62 = _mm_loadu_si128(a1 + 6);
  v64 = (_BYTE **)_mm_loadu_si128(a1 + 8).m128i_u64[0];
  v67 = v3;
  v65 = (_BYTE **)a2;
  v63 = _mm_loadu_si128(a1 + 7);
  v66 = _mm_loadu_si128(a1 + 9);
  v4 = sub_B45210(a2);
  v5 = sub_100EA30(*(__int64 **)(a2 - 64), *(_BYTE **)(a2 - 32), v4, &v62, 0, 1);
  if ( v5 )
  {
    if ( *(_QWORD *)(a2 + 16) )
    {
      v6 = (__int64)v5;
      sub_10A5FE0(a1[2].m128i_i64[1], a2);
      if ( a2 == v6 )
      {
        v6 = sub_ACADE0(*(__int64 ***)(a2 + 8));
        if ( *(_QWORD *)(v6 + 16) )
          goto LABEL_5;
      }
      else if ( *(_QWORD *)(v6 + 16) )
      {
LABEL_5:
        sub_BD84D0(a2, v6);
        return v2;
      }
      if ( *(_BYTE *)v6 > 0x1Cu && (*(_BYTE *)(v6 + 7) & 0x10) == 0 && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
        sub_BD6B90((unsigned __int8 *)v6, (unsigned __int8 *)a2);
      goto LABEL_5;
    }
    return 0;
  }
  if ( (unsigned __int8)sub_F29CA0(a1, (unsigned __int8 *)a2) )
    return v2;
  v8 = (__int64)sub_F0F270((__int64)a1, (unsigned __int8 *)a2);
  if ( v8 )
    return (unsigned __int8 *)v8;
  v8 = sub_F11DB0(a1->m128i_i64, (unsigned __int8 *)a2);
  if ( v8 )
    return (unsigned __int8 *)v8;
  v8 = (__int64)sub_F28360((__int64)a1, (_BYTE *)a2, v9, v10, v11, v12);
  if ( v8 )
    return (unsigned __int8 *)v8;
  v16 = *(_BYTE *)a2 == 43;
  v62.m128i_i64[0] = (__int64)&v53;
  v62.m128i_i64[1] = (__int64)&v54;
  if ( v16 )
  {
    v22 = sub_995E90(&v62, *(_QWORD *)(a2 - 64), v13, (__int64)v14, v15);
    v26 = *(_QWORD *)(a2 - 32);
    if ( v22 && v26 )
    {
      *(_QWORD *)v62.m128i_i64[1] = v26;
    }
    else
    {
      if ( !(unsigned __int8)sub_995E90(&v62, v26, v23, v24, v25) )
        goto LABEL_20;
      v31 = *((_QWORD *)v2 - 8);
      if ( !v31 )
        goto LABEL_20;
      *(_QWORD *)v62.m128i_i64[1] = v31;
    }
    LOWORD(v64) = 257;
    v27 = (unsigned __int8 *)sub_B504D0(16, (__int64)v54, (__int64)v53, (__int64)&v62, 0, 0);
    sub_B45260(v27, (__int64)v2, 1);
    return v27;
  }
LABEL_20:
  v62.m128i_i64[0] = (__int64)&v53;
  v62.m128i_i64[1] = (__int64)&v54;
  v63.m128i_i64[0] = (__int64)&v55;
  if ( (unsigned __int8)sub_10A9380(&v62, 14, v2, v14, v15) )
  {
    v19 = (unsigned int **)a1[2].m128i_i64[0];
    LOWORD(v64) = 257;
    sub_10A0170((__int64)&v59, (__int64)v2);
    v20 = sub_A826E0(v19, v53, v54, (__int64)v59, (__int64)&v62, 0);
LABEL_22:
    LOWORD(v64) = 257;
    return sub_109FE60(16, v55, v20, (__int64)v2, (__int64)&v62, v21, 0, 0);
  }
  v59 = &v53;
  v60 = &v54;
  v61 = &v55;
  if ( (unsigned __int8)sub_10A94F0((__int64)&v59, 14, v2, v17, v18)
    || (v62.m128i_i64[0] = (__int64)&v53,
        v62.m128i_i64[1] = (__int64)&v54,
        v63.m128i_i64[0] = (__int64)&v55,
        (unsigned __int8)sub_10A95E0(&v62, 14, v2, v28, v29)) )
  {
    v30 = (unsigned int **)a1[2].m128i_i64[0];
    LOWORD(v64) = 257;
    sub_10A0170((__int64)&v59, (__int64)v2);
    v20 = sub_A82920(v30, v53, v54, (__int64)v59, (__int64)&v62, 0);
    goto LABEL_22;
  }
  v8 = (__int64)sub_F18290(a1, v2);
  if ( v8 )
    return (unsigned __int8 *)v8;
  v32 = (unsigned __int8 *)*((_QWORD *)v2 - 4);
  v51 = *((_QWORD *)v2 - 8);
  v33 = sub_F0D870(a1, v2, v51, (__int64)v32);
  if ( v33 )
    return sub_F162A0((__int64)a1, (__int64)v2, (__int64)v33);
  if ( sub_B451B0((__int64)v2) && sub_B451E0((__int64)v2) )
  {
    v8 = (__int64)sub_10B5C50(v2, a1[2].m128i_i64[0]);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v8 = (__int64)sub_10B70C0((__int64)a1, (char *)v2);
    if ( v8 )
      return (unsigned __int8 *)v8;
    v62.m128i_i32[0] = 389;
    v62.m128i_i32[2] = 0;
    v63.m128i_i64[0] = 0;
    v63.m128i_i32[2] = 1;
    v64 = &v53;
    v65 = &v54;
    if ( (unsigned __int8)sub_10A96D0((__int64)&v62, 14, v2) )
    {
      v35 = a1[2].m128i_i64[0];
      LOWORD(v64) = 257;
      sub_10A0170((__int64)&v58, (__int64)v2);
      v59 = (_BYTE **)v54;
LABEL_45:
      v60 = (_BYTE **)v53;
      v57 = *((_QWORD *)v53 + 1);
      v36 = sub_B33D10(v35, 0x185u, (__int64)&v57, 1, (int)&v59, 2, v58, (__int64)&v62);
      return sub_F162A0((__int64)a1, (__int64)v2, v36);
    }
    v62.m128i_i32[0] = 389;
    v62.m128i_i32[2] = 0;
    v63.m128i_i64[0] = (__int64)&v56;
    v63.m128i_i8[8] = 0;
    LODWORD(v64) = 1;
    v65 = &v53;
    if ( (unsigned __int8)sub_10A4BC0((__int64)&v62, v51) )
    {
      v37 = *v32;
      if ( (_BYTE)v37 == 18
        || (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v32 + 1) + 8LL) - 17 <= 1
        && (unsigned __int8)v37 <= 0x15u
        && (v42 = sub_AD7630((__int64)v32, 0, v37), (v32 = v42) != 0)
        && *v42 == 18 )
      {
        v38 = sub_C33340();
        v39 = v56;
        v52 = v38;
        v40 = (__int64 *)(v32 + 24);
        if ( *((void **)v32 + 3) == v38 )
          sub_C3C790(&v62, (_QWORD **)v40);
        else
          sub_C33EB0(&v62, v40);
        if ( (void *)v62.m128i_i64[0] == v52 )
          sub_C3D800(v62.m128i_i64, v39, 1u);
        else
          sub_C3ADF0((__int64)&v62, v39, 1);
        v41 = sub_AD8F10(*((_QWORD *)v2 + 1), v62.m128i_i64);
        sub_91D830(&v62);
        v35 = a1[2].m128i_i64[0];
        LOWORD(v64) = 257;
        sub_10A0170((__int64)&v58, (__int64)v2);
        v59 = (_BYTE **)v41;
        goto LABEL_45;
      }
    }
    v62.m128i_i64[0] = (__int64)&v53;
    v62.m128i_i64[1] = (__int64)&v59;
    v63.m128i_i64[1] = (__int64)&v53;
    if ( sub_10A9B60(&v62, 14, v2)
      && (v45 = a1[5].m128i_i64[1],
          v46 = sub_AD8DD0(*((_QWORD *)v2 + 1), 1.0),
          (v47 = sub_96E6C0(0xEu, (__int64)v59, v46, v45)) != 0) )
    {
      LOWORD(v64) = 257;
      return sub_109FE60(18, (__int64)v53, v47, (__int64)v2, (__int64)&v62, v48, 0, 0);
    }
    else
    {
      v62.m128i_i64[0] = (__int64)&v53;
      v62.m128i_i64[1] = (__int64)&v54;
      v63.m128i_i64[0] = (__int64)&v53;
      v63.m128i_i64[1] = (__int64)&v55;
      if ( (unsigned __int8)sub_10A9C50(&v62, 14, v2, v43, v44) )
      {
        LOWORD(v64) = 257;
        return sub_109FE60(16, v55, (__int64)v54, (__int64)v2, (__int64)&v62, v49, 0, 0);
      }
      else
      {
        v62.m128i_i64[1] = 0;
        v62.m128i_i64[0] = a1[2].m128i_i64[0];
        if ( (unsigned __int8)(*(_BYTE *)(*((_QWORD *)v2 + 1) + 8LL) - 17) <= 1u )
          goto LABEL_38;
        v50 = sub_109F890(v62.m128i_i64, (__int64)v2);
        if ( !v50 )
          goto LABEL_38;
        return sub_F162A0((__int64)a1, (__int64)v2, (__int64)v50);
      }
    }
  }
LABEL_38:
  v66.m128i_i64[1] = (__int64)&v53;
  v63.m128i_i64[0] = (__int64)&v53;
  v62.m128i_i32[0] = 235;
  v62.m128i_i32[2] = 0;
  v63.m128i_i32[2] = 1;
  v64 = &v54;
  LODWORD(v65) = 246;
  v66.m128i_i32[0] = 0;
  LODWORD(v67) = 1;
  v68 = &v54;
  v69 = 246;
  v70 = 0;
  v71 = &v54;
  v72 = 1;
  v73 = &v53;
  if ( !sub_10AC220((__int64)&v62, 14, v2) )
    return 0;
  LOWORD(v64) = 257;
  v2 = sub_109FE60(14, (__int64)v53, (__int64)v54, (__int64)v2, (__int64)&v62, v34, 0, 0);
  if ( !sub_B451C0((__int64)v2) )
    sub_B44F10((__int64)v2, 0);
  return v2;
}
