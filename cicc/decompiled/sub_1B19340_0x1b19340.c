// Function: sub_1B19340
// Address: 0x1b19340
//
_QWORD *__fastcall sub_1B19340(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __m128i a6,
        __m128i a7,
        double a8)
{
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // eax
  _QWORD *v16; // r13
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // rax
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r12
  __int64 v24; // rcx
  __int64 v25; // rax
  _BYTE *v26; // rdi
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rax
  _BYTE *v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // rbx
  _QWORD *v33; // r12
  __int64 v34; // rax
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  int v44; // r15d
  __int64 v45; // rax
  __int64 *v46; // r15
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 *v55; // rsi
  int v56; // edi
  __int64 v57; // rax
  unsigned __int64 *v58; // rbx
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rsi
  __int64 v62; // rsi
  unsigned __int8 *v63; // rsi
  __int64 v64; // [rsp+8h] [rbp-238h]
  __int64 v65; // [rsp+10h] [rbp-230h]
  __int64 v66; // [rsp+28h] [rbp-218h] BYREF
  __int64 v67[2]; // [rsp+30h] [rbp-210h] BYREF
  __int16 v68; // [rsp+40h] [rbp-200h]
  __int64 *v69; // [rsp+50h] [rbp-1F0h] BYREF
  __int64 v70; // [rsp+58h] [rbp-1E8h]
  __int64 v71; // [rsp+60h] [rbp-1E0h] BYREF
  __int64 v72; // [rsp+68h] [rbp-1D8h]
  _QWORD v73[4]; // [rsp+70h] [rbp-1D0h] BYREF
  _QWORD *v74; // [rsp+90h] [rbp-1B0h]
  __int64 v75; // [rsp+98h] [rbp-1A8h]
  unsigned int v76; // [rsp+A0h] [rbp-1A0h]
  __int64 v77; // [rsp+A8h] [rbp-198h]
  __int64 v78; // [rsp+B0h] [rbp-190h]
  __int64 v79; // [rsp+B8h] [rbp-188h]
  __int64 v80; // [rsp+C0h] [rbp-180h]
  __int64 v81; // [rsp+C8h] [rbp-178h]
  __int64 v82; // [rsp+D0h] [rbp-170h]
  __int64 v83; // [rsp+D8h] [rbp-168h]
  __int64 v84; // [rsp+E0h] [rbp-160h]
  __int64 v85; // [rsp+E8h] [rbp-158h]
  __int64 v86; // [rsp+F0h] [rbp-150h]
  __int64 v87; // [rsp+F8h] [rbp-148h]
  int v88; // [rsp+100h] [rbp-140h]
  __int64 v89; // [rsp+108h] [rbp-138h]
  _BYTE *v90; // [rsp+110h] [rbp-130h]
  _BYTE *v91; // [rsp+118h] [rbp-128h]
  __int64 v92; // [rsp+120h] [rbp-120h]
  int v93; // [rsp+128h] [rbp-118h]
  _BYTE v94[16]; // [rsp+130h] [rbp-110h] BYREF
  __int64 v95; // [rsp+140h] [rbp-100h]
  __int64 v96; // [rsp+148h] [rbp-F8h]
  __int64 v97; // [rsp+150h] [rbp-F0h]
  __int64 v98; // [rsp+158h] [rbp-E8h]
  __int64 v99; // [rsp+160h] [rbp-E0h]
  __int64 v100; // [rsp+168h] [rbp-D8h]
  __int16 v101; // [rsp+170h] [rbp-D0h]
  __int64 v102[5]; // [rsp+178h] [rbp-C8h] BYREF
  int v103; // [rsp+1A0h] [rbp-A0h]
  __int64 v104; // [rsp+1A8h] [rbp-98h]
  __int64 v105; // [rsp+1B0h] [rbp-90h]
  __int64 v106; // [rsp+1B8h] [rbp-88h]
  _BYTE *v107; // [rsp+1C0h] [rbp-80h]
  __int64 v108; // [rsp+1C8h] [rbp-78h]
  _BYTE v109[112]; // [rsp+1D0h] [rbp-70h] BYREF

  v73[2] = "induction";
  v73[0] = a4;
  v73[1] = a5;
  v90 = v94;
  v91 = v94;
  v73[3] = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v92 = 2;
  v93 = 0;
  v95 = 0;
  v96 = 0;
  v13 = a4[3];
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 1;
  v14 = sub_15E0530(v13);
  v106 = a5;
  v102[3] = v14;
  v107 = v109;
  v108 = 0x800000000LL;
  v15 = *(_DWORD *)(a1 + 24);
  memset(v102, 0, 24);
  v102[4] = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  if ( v15 == 2 )
  {
    v64 = *(_QWORD *)(a1 + 32);
    v71 = sub_146F1B0((__int64)a4, a3);
    v69 = &v71;
    v72 = v64;
    v70 = 0x200000002LL;
    v27 = sub_147EE30(a4, &v69, 0, 0, a6, a7);
    if ( v69 != &v71 )
      _libc_free((unsigned __int64)v69);
    v28 = *(_QWORD *)(a2 + 16);
    if ( v28 )
      v28 -= 24;
    v29 = sub_38767A0(v73, v27, *(_QWORD *)a3, v28);
    v30 = *(_BYTE **)(a1 + 16);
    LOWORD(v71) = 257;
    v31 = sub_12815B0((__int64 *)a2, 0, v30, v29, (__int64)&v69);
    v26 = v107;
    v16 = (_QWORD *)v31;
    goto LABEL_22;
  }
  if ( v15 > 2 )
  {
    v36 = *(_QWORD *)(a1 + 32);
    if ( !v36 )
      BUG();
    v37 = *(_QWORD *)(v36 - 8);
    v68 = 257;
    if ( *(_BYTE *)(v37 + 16) > 0x10u
      || *(_BYTE *)(a3 + 16) > 0x10u
      || (v38 = sub_15A2A30(
                  (__int64 *)0x10,
                  (__int64 *)v37,
                  a3,
                  0,
                  0,
                  *(double *)a6.m128i_i64,
                  *(double *)a7.m128i_i64,
                  a8)) == 0 )
    {
      LOWORD(v71) = 257;
      v42 = sub_15FB440(16, (__int64 *)v37, a3, (__int64)&v69, 0);
      v43 = *(_QWORD *)(a2 + 32);
      v44 = *(_DWORD *)(a2 + 40);
      v38 = v42;
      if ( v43 )
        sub_1625C10(v42, 3, v43);
      sub_15F2440(v38, v44);
      v45 = *(_QWORD *)(a2 + 8);
      if ( v45 )
      {
        v46 = *(__int64 **)(a2 + 16);
        sub_157E9D0(v45 + 40, v38);
        v47 = *(_QWORD *)(v38 + 24);
        v48 = *v46;
        *(_QWORD *)(v38 + 32) = v46;
        v48 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v38 + 24) = v48 | v47 & 7;
        *(_QWORD *)(v48 + 8) = v38 + 24;
        *v46 = *v46 & 7 | (v38 + 24);
      }
      sub_164B780(v38, v67);
      v49 = *(_QWORD *)a2;
      if ( *(_QWORD *)a2 )
      {
        v69 = *(__int64 **)a2;
        sub_1623A60((__int64)&v69, v49, 2);
        v50 = *(_QWORD *)(v38 + 48);
        if ( v50 )
          sub_161E7C0(v38 + 48, v50);
        v51 = (unsigned __int8 *)v69;
        *(_QWORD *)(v38 + 48) = v69;
        if ( v51 )
          sub_1623210((__int64)&v69, v51, v38 + 48);
      }
    }
    if ( *(_BYTE *)(v38 + 16) > 0x17u )
      sub_15F2440(v38, -1);
    v69 = (__int64 *)"induction";
    v39 = *(_QWORD *)(a1 + 40);
    v40 = *(_QWORD *)(a1 + 16);
    LOWORD(v71) = 259;
    v41 = sub_1904E90(
            a2,
            (unsigned int)*(unsigned __int8 *)(v39 + 16) - 24,
            v40,
            v38,
            (__int64 *)&v69,
            0,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64,
            a8);
    v16 = (_QWORD *)v41;
    if ( *(_BYTE *)(v41 + 16) > 0x17u )
      sub_15F2440(v41, -1);
    goto LABEL_49;
  }
  v16 = 0;
  if ( !v15 )
    goto LABEL_24;
  if ( !sub_1B16970(a1) )
    goto LABEL_7;
  v17 = sub_1B16970(a1);
  v18 = *(_DWORD *)(v17 + 32);
  if ( v18 <= 0x40 )
  {
    if ( *(_QWORD *)(v17 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) )
      goto LABEL_7;
  }
  else if ( v18 != (unsigned int)sub_16A58F0(v17 + 24) )
  {
LABEL_7:
    if ( !sub_1B16970(a1) )
    {
LABEL_10:
      v21 = sub_146F1B0((__int64)a4, a3);
      v22 = *(_QWORD *)(a1 + 32);
      v72 = v21;
      v71 = v22;
      v69 = &v71;
      v70 = 0x200000002LL;
      v65 = sub_147EE30(a4, &v69, 0, 0, a6, a7);
      if ( v69 != &v71 )
        _libc_free((unsigned __int64)v69);
      v71 = sub_146F1B0((__int64)a4, *(_QWORD *)(a1 + 16));
      v69 = &v71;
      v72 = v65;
      v70 = 0x200000002LL;
      v23 = sub_147DD40((__int64)a4, (__int64 *)&v69, 0, 0, a6, a7);
      if ( v69 != &v71 )
        _libc_free((unsigned __int64)v69);
      v24 = *(_QWORD *)(a2 + 16);
      if ( v24 )
        v24 -= 24;
      v25 = sub_38767A0(v73, v23, **(_QWORD **)(a1 + 16), v24);
      v26 = v107;
      v16 = (_QWORD *)v25;
      goto LABEL_22;
    }
    v19 = sub_1B16970(a1);
    v20 = *(_DWORD *)(v19 + 32);
    if ( v20 <= 0x40 )
    {
      if ( *(_QWORD *)(v19 + 24) != 1 )
        goto LABEL_10;
    }
    else if ( (unsigned int)sub_16A57B0(v19 + 24) != v20 - 1 )
    {
      goto LABEL_10;
    }
    v53 = *(_QWORD *)(a1 + 16);
    v68 = 257;
    if ( *(_BYTE *)(v53 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    {
      v16 = (_QWORD *)sub_15A2B30((__int64 *)v53, a3, 0, 0, *(double *)a6.m128i_i64, *(double *)a7.m128i_i64, a8);
      goto LABEL_49;
    }
    v54 = a3;
    v55 = (__int64 *)v53;
    LOWORD(v71) = 257;
    v56 = 11;
LABEL_68:
    v16 = (_QWORD *)sub_15FB440(v56, v55, v54, (__int64)&v69, 0);
    v57 = *(_QWORD *)(a2 + 8);
    if ( v57 )
    {
      v58 = *(unsigned __int64 **)(a2 + 16);
      sub_157E9D0(v57 + 40, (__int64)v16);
      v59 = v16[3];
      v60 = *v58;
      v16[4] = v58;
      v60 &= 0xFFFFFFFFFFFFFFF8LL;
      v16[3] = v60 | v59 & 7;
      *(_QWORD *)(v60 + 8) = v16 + 3;
      *v58 = *v58 & 7 | (unsigned __int64)(v16 + 3);
    }
    sub_164B780((__int64)v16, v67);
    v61 = *(_QWORD *)a2;
    if ( *(_QWORD *)a2 )
    {
      v66 = *(_QWORD *)a2;
      sub_1623A60((__int64)&v66, v61, 2);
      v62 = v16[6];
      if ( v62 )
        sub_161E7C0((__int64)(v16 + 6), v62);
      v63 = (unsigned __int8 *)v66;
      v16[6] = v66;
      if ( v63 )
        sub_1623210((__int64)&v66, v63, (__int64)(v16 + 6));
    }
    goto LABEL_49;
  }
  v52 = *(_QWORD *)(a1 + 16);
  v68 = 257;
  if ( *(_BYTE *)(v52 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v54 = a3;
    v55 = (__int64 *)v52;
    LOWORD(v71) = 257;
    v56 = 13;
    goto LABEL_68;
  }
  v16 = (_QWORD *)sub_15A2B60((__int64 *)v52, a3, 0, 0, *(double *)a6.m128i_i64, *(double *)a7.m128i_i64, a8);
LABEL_49:
  v26 = v107;
LABEL_22:
  if ( v26 != v109 )
    _libc_free((unsigned __int64)v26);
LABEL_24:
  if ( v102[0] )
    sub_161E7C0((__int64)v102, v102[0]);
  j___libc_free_0(v98);
  if ( v91 != v90 )
    _libc_free((unsigned __int64)v91);
  j___libc_free_0(v86);
  j___libc_free_0(v82);
  j___libc_free_0(v78);
  if ( v76 )
  {
    v32 = v74;
    v33 = &v74[5 * v76];
    do
    {
      while ( *v32 == -8 )
      {
        if ( v32[1] != -8 )
          goto LABEL_31;
        v32 += 5;
        if ( v33 == v32 )
          goto LABEL_38;
      }
      if ( *v32 != -16 || v32[1] != -16 )
      {
LABEL_31:
        v34 = v32[4];
        if ( v34 != 0 && v34 != -8 && v34 != -16 )
          sub_1649B30(v32 + 2);
      }
      v32 += 5;
    }
    while ( v33 != v32 );
  }
LABEL_38:
  j___libc_free_0(v74);
  return v16;
}
