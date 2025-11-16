// Function: sub_171BFC0
// Address: 0x171bfc0
//
unsigned __int64 __fastcall sub_171BFC0(__int64 *a1, __int64 a2, __m128i a3, double a4, double a5)
{
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r8
  int v10; // r9d
  __int64 v11; // rcx
  unsigned int v12; // r13d
  unsigned __int64 v13; // r13
  int v15; // eax
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rax
  unsigned __int64 **v20; // rdi
  __int64 *v21; // rax
  __int64 *v22; // rax
  _QWORD *v23; // rdx
  __int64 v24; // rax
  unsigned __int8 v25; // si
  __int64 v26; // rdx
  unsigned __int8 v27; // di
  char v28; // bl
  __int64 *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r14
  __int64 *v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // r15
  unsigned __int8 v36; // al
  __int64 v37; // rdi
  void *v38; // r13
  char v39; // al
  char v40; // al
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  int v45; // r14d
  __int64 v46; // rax
  unsigned __int64 *v47; // r14
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rsi
  __int64 v52; // rsi
  unsigned __int8 *v53; // rsi
  __int64 v54; // rdx
  int v55; // [rsp+10h] [rbp-1C0h]
  unsigned int v56; // [rsp+20h] [rbp-1B0h]
  __int64 v57; // [rsp+20h] [rbp-1B0h]
  unsigned int v58; // [rsp+20h] [rbp-1B0h]
  unsigned int v59; // [rsp+28h] [rbp-1A8h]
  int v60; // [rsp+28h] [rbp-1A8h]
  int v61; // [rsp+28h] [rbp-1A8h]
  __int64 v62[2]; // [rsp+30h] [rbp-1A0h] BYREF
  __int16 v63; // [rsp+40h] [rbp-190h]
  unsigned __int64 v64; // [rsp+50h] [rbp-180h] BYREF
  int v65; // [rsp+58h] [rbp-178h]
  __int64 v66[3]; // [rsp+68h] [rbp-168h] BYREF
  __int64 v67; // [rsp+80h] [rbp-150h] BYREF
  int v68; // [rsp+88h] [rbp-148h]
  __int64 v69[3]; // [rsp+98h] [rbp-138h] BYREF
  __int64 v70; // [rsp+B0h] [rbp-120h] BYREF
  int v71; // [rsp+B8h] [rbp-118h]
  __int64 v72[3]; // [rsp+C8h] [rbp-108h] BYREF
  __int64 v73; // [rsp+E0h] [rbp-F0h] BYREF
  int v74; // [rsp+E8h] [rbp-E8h]
  __int64 v75[3]; // [rsp+F8h] [rbp-D8h] BYREF
  __int64 v76; // [rsp+110h] [rbp-C0h] BYREF
  int v77; // [rsp+118h] [rbp-B8h]
  __int64 v78[3]; // [rsp+128h] [rbp-A8h] BYREF
  __int64 v79; // [rsp+140h] [rbp-90h] BYREF
  int v80; // [rsp+148h] [rbp-88h]
  __int64 v81[3]; // [rsp+158h] [rbp-78h] BYREF
  unsigned __int64 **v82; // [rsp+170h] [rbp-60h] BYREF
  __int64 v83; // [rsp+178h] [rbp-58h]
  unsigned __int64 *v84; // [rsp+180h] [rbp-50h] BYREF
  __int64 *v85; // [rsp+188h] [rbp-48h]
  __int64 *v86; // [rsp+190h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    return 0;
  a1[1] = a2;
  v64 = 0;
  v65 = 0;
  v67 = 0;
  v68 = 0;
  v70 = 0;
  v71 = 0;
  v73 = 0;
  v74 = 0;
  v76 = 0;
  v77 = 0;
  v79 = 0;
  v80 = 0;
  v7 = sub_171A810(a2, (__int64)&v64, (__int64)&v67, *(double *)a3.m128i_i64, a4, a5);
  v11 = 0;
  v12 = v7;
  if ( v64 )
    v11 = (unsigned int)sub_171BF50((__int64)&v64, (__int64)&v70, (__int64)&v73, *(double *)a3.m128i_i64, a4, a5);
  if ( v12 != 2 )
  {
    if ( !(_BYTE)v65 )
    {
      v13 = v64;
      if ( HIWORD(v65) == 1 )
        goto LABEL_7;
    }
LABEL_6:
    v13 = 0;
    goto LABEL_7;
  }
  if ( !v67 )
  {
LABEL_38:
    if ( (_DWORD)v11 )
    {
      v84 = (unsigned __int64 *)&v67;
      v85 = &v70;
      v82 = &v84;
      v83 = 0x400000002LL;
      if ( (_DWORD)v11 == 2 )
      {
        v62[0] = (__int64)&v73;
        sub_1718F60((__int64)&v82, v62, v8, v11, v9, v10);
      }
      v22 = sub_171AB60((__int64)a1, (__int64)&v82, 1u, a3, a4, a5, v11, v9, v10);
      v20 = v82;
      v13 = (unsigned __int64)v22;
      if ( v22 )
        goto LABEL_42;
      if ( v82 != &v84 )
        _libc_free((unsigned __int64)v82);
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v23 = *(_QWORD **)(a2 - 8);
    else
      v23 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v24 = *v23;
    v25 = *(_BYTE *)(*v23 + 16LL);
    if ( v25 <= 0x17u )
      goto LABEL_6;
    v26 = v23[3];
    v27 = *(_BYTE *)(v26 + 16);
    if ( v27 <= 0x17u || v25 != v27 )
      goto LABEL_6;
    if ( v25 == 40 )
    {
      v28 = 1;
    }
    else
    {
      v28 = 0;
      if ( v25 != 43 )
        goto LABEL_6;
    }
    if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
      v29 = *(__int64 **)(v24 - 8);
    else
      v29 = (__int64 *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF));
    v30 = *v29;
    v31 = v29[3];
    if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
      v32 = *(__int64 **)(v26 - 8);
    else
      v32 = (__int64 *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
    v33 = *v32;
    v34 = v32[3];
    if ( v28 )
    {
      if ( v30 == v33 || v30 == v34 )
      {
        if ( !v30 )
          goto LABEL_6;
        v54 = v30;
        v30 = v31;
        v31 = v54;
      }
      else if ( v31 != v34 && v31 != v33 || !v31 )
      {
        goto LABEL_6;
      }
      if ( v31 == v33 )
        v33 = v34;
    }
    else if ( !v31 || v31 != v34 )
    {
      goto LABEL_6;
    }
    v60 = sub_15F24E0(a2);
    v61 = v60 & sub_15F24E0(a2);
    if ( *(_BYTE *)(a2 + 16) == 36 )
      v35 = sub_1719710(a1, v30, v33, *(double *)a3.m128i_i64, a4, a5);
    else
      v35 = sub_1719550(a1, v30, v33, *(double *)a3.m128i_i64, a4, a5);
    v36 = *(_BYTE *)(v35 + 16);
    if ( v36 == 14 )
    {
      v37 = v35 + 32;
      v38 = sub_16982C0();
      if ( *(void **)(v35 + 32) == v38 )
        v39 = sub_16A0F40(v37, v30, *(double *)a3.m128i_i64, a4, a5);
      else
        v39 = sub_16984B0(v37);
      if ( v39 )
        goto LABEL_6;
      if ( v38 == *(void **)(v35 + 32) )
      {
        v40 = *(_BYTE *)(*(_QWORD *)(v35 + 40) + 26LL) & 7;
        if ( v40 == 1 )
          goto LABEL_6;
      }
      else
      {
        v40 = *(_BYTE *)(v35 + 50) & 7;
        if ( v40 == 1 )
          goto LABEL_6;
      }
      if ( !v40 || v40 == 3 )
        goto LABEL_6;
    }
    else if ( v36 > 0x17u )
    {
      sub_15F2440(v35, v61);
    }
    if ( v28 )
    {
      v13 = sub_1719390(a1, v31, v35, *(double *)a3.m128i_i64, a4, a5);
      if ( *(_BYTE *)(v13 + 16) <= 0x17u )
        goto LABEL_7;
    }
    else
    {
      v41 = *a1;
      v63 = 257;
      if ( *(_BYTE *)(v35 + 16) > 0x10u
        || *(_BYTE *)(v31 + 16) > 0x10u
        || (v57 = sub_15A2A30((__int64 *)0x13, (__int64 *)v35, v31, 0, 0, *(double *)a3.m128i_i64, a4, a5),
            (v13 = sub_14DBA30(v57, *(_QWORD *)(v41 + 96), 0)) == 0)
        && (v13 = v57) == 0 )
      {
        LOWORD(v84) = 257;
        v43 = sub_15FB440(19, (__int64 *)v35, v31, (__int64)&v82, 0);
        v44 = *(_QWORD *)(v41 + 32);
        v45 = *(_DWORD *)(v41 + 40);
        v13 = v43;
        if ( v44 )
          sub_1625C10(v43, 3, v44);
        sub_15F2440(v13, v45);
        v46 = *(_QWORD *)(v41 + 8);
        if ( v46 )
        {
          v47 = *(unsigned __int64 **)(v41 + 16);
          sub_157E9D0(v46 + 40, v13);
          v48 = *(_QWORD *)(v13 + 24);
          v49 = *v47;
          *(_QWORD *)(v13 + 32) = v47;
          v49 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v13 + 24) = v49 | v48 & 7;
          *(_QWORD *)(v49 + 8) = v13 + 24;
          *v47 = *v47 & 7 | (v13 + 24);
        }
        sub_164B780(v13, v62);
        v82 = (unsigned __int64 **)v13;
        if ( !*(_QWORD *)(v41 + 80) )
          sub_4263D6(v13, v62, v50);
        (*(void (__fastcall **)(__int64, unsigned __int64 ***))(v41 + 88))(v41 + 64, &v82);
        v51 = *(_QWORD *)v41;
        if ( *(_QWORD *)v41 )
        {
          v82 = *(unsigned __int64 ***)v41;
          sub_1623A60((__int64)&v82, v51, 2);
          v52 = *(_QWORD *)(v13 + 48);
          if ( v52 )
            sub_161E7C0(v13 + 48, v52);
          v53 = (unsigned __int8 *)v82;
          *(_QWORD *)(v13 + 48) = v82;
          if ( v53 )
            sub_1623210((__int64)&v82, v53, v13 + 48);
        }
      }
      if ( *(_BYTE *)(v13 + 16) <= 0x17u || (sub_1718FD0((__int64)a1, v13), *(_BYTE *)(v13 + 16) <= 0x17u) )
      {
LABEL_7:
        if ( !BYTE1(v80) )
          goto LABEL_8;
        goto LABEL_44;
      }
    }
    sub_15F2440(v13, v61);
    goto LABEL_7;
  }
  v56 = v11;
  v15 = sub_171BF50((__int64)&v67, (__int64)&v76, (__int64)&v79, *(double *)a3.m128i_i64, a4, a5);
  v11 = v56;
  v10 = v15;
  if ( v56 )
  {
    if ( !v15 )
      goto LABEL_38;
    v84 = (unsigned __int64 *)&v70;
    v82 = &v84;
    v85 = &v76;
    v83 = 0x400000002LL;
    if ( v56 == 2 )
    {
      LODWORD(v83) = 3;
      v86 = &v73;
    }
    if ( v15 == 2 )
    {
      (&v84)[(unsigned int)v83] = (unsigned __int64 *)&v79;
      LODWORD(v83) = v83 + 1;
    }
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v16 = *(_QWORD **)(a2 - 8);
    else
      v16 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(*v16 + 16LL) > 0x10u
      && (v17 = *(_QWORD *)(*v16 + 8LL)) != 0
      && !*(_QWORD *)(v17 + 8)
      && (v18 = v16[3], *(_BYTE *)(v18 + 16) > 0x10u)
      && (v42 = *(_QWORD *)(v18 + 8)) != 0 )
    {
      if ( *(_QWORD *)(v42 + 8) )
        v12 = 1;
    }
    else
    {
      v12 = 1;
    }
    v55 = v10;
    v19 = sub_171AB60((__int64)a1, (__int64)&v82, v12, a3, a4, a5, v56, v9, v10);
    v20 = v82;
    v11 = v56;
    v13 = (unsigned __int64)v19;
    v10 = v55;
    if ( v19 )
      goto LABEL_42;
    if ( v82 != &v84 )
    {
      _libc_free((unsigned __int64)v82);
      v11 = v56;
      v10 = v55;
    }
  }
  else if ( !v15 )
  {
    goto LABEL_38;
  }
  v82 = &v84;
  v84 = &v64;
  v85 = &v76;
  v83 = 0x400000002LL;
  if ( v10 == 2 )
  {
    v58 = v11;
    v62[0] = (__int64)&v79;
    sub_1718F60((__int64)&v82, v62, v8, v11, v9, 2);
    v11 = v58;
  }
  v59 = v11;
  v21 = sub_171AB60((__int64)a1, (__int64)&v82, 1u, a3, a4, a5, v11, v9, v10);
  v20 = v82;
  v11 = v59;
  v13 = (unsigned __int64)v21;
  if ( !v21 )
  {
    if ( v82 != &v84 )
    {
      _libc_free((unsigned __int64)v82);
      v11 = v59;
    }
    goto LABEL_38;
  }
LABEL_42:
  if ( v20 == &v84 )
    goto LABEL_7;
  _libc_free((unsigned __int64)v20);
  if ( !BYTE1(v80) )
  {
LABEL_8:
    if ( !BYTE1(v77) )
      goto LABEL_9;
    goto LABEL_45;
  }
LABEL_44:
  sub_127D120(v81);
  if ( !BYTE1(v77) )
  {
LABEL_9:
    if ( !BYTE1(v74) )
      goto LABEL_10;
    goto LABEL_46;
  }
LABEL_45:
  sub_127D120(v78);
  if ( !BYTE1(v74) )
  {
LABEL_10:
    if ( !BYTE1(v71) )
      goto LABEL_11;
    goto LABEL_47;
  }
LABEL_46:
  sub_127D120(v75);
  if ( !BYTE1(v71) )
  {
LABEL_11:
    if ( !BYTE1(v68) )
      goto LABEL_12;
LABEL_48:
    sub_127D120(v69);
    if ( !BYTE1(v65) )
      return v13;
LABEL_49:
    sub_127D120(v66);
    return v13;
  }
LABEL_47:
  sub_127D120(v72);
  if ( BYTE1(v68) )
    goto LABEL_48;
LABEL_12:
  if ( BYTE1(v65) )
    goto LABEL_49;
  return v13;
}
