// Function: sub_258AB20
// Address: 0x258ab20
//
_BOOL8 __fastcall sub_258AB20(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rbx
  unsigned __int8 v5; // cl
  unsigned __int8 *v6; // r15
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __m128i v9; // rax
  __m128i v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // eax
  _BOOL4 v16; // r12d
  char v18; // al
  int v19; // eax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __m128i v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __m128i v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  char v39; // bl
  _QWORD *v40; // rbx
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __m128i v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __m128i v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // r11
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  char v61; // r12
  char v62; // r12
  __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rsi
  __m128i v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // r12
  __int64 v76; // [rsp-10h] [rbp-1C0h]
  __int64 v77; // [rsp-10h] [rbp-1C0h]
  __int64 v78; // [rsp-10h] [rbp-1C0h]
  __int64 v79; // [rsp-10h] [rbp-1C0h]
  __int64 v80; // [rsp-8h] [rbp-1B8h]
  unsigned __int64 v81; // [rsp+8h] [rbp-1A8h]
  __int64 v82; // [rsp+8h] [rbp-1A8h]
  unsigned __int64 v83; // [rsp+8h] [rbp-1A8h]
  unsigned __int64 v84; // [rsp+10h] [rbp-1A0h]
  __int64 v85; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v86; // [rsp+10h] [rbp-1A0h]
  __int64 v87; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v88; // [rsp+20h] [rbp-190h]
  unsigned __int64 v89; // [rsp+20h] [rbp-190h]
  __int64 v90; // [rsp+20h] [rbp-190h]
  __int64 v91; // [rsp+28h] [rbp-188h]
  char v92; // [rsp+38h] [rbp-178h]
  char v93; // [rsp+4Fh] [rbp-161h] BYREF
  unsigned __int64 v94; // [rsp+50h] [rbp-160h] BYREF
  __int64 v95; // [rsp+58h] [rbp-158h]
  unsigned __int64 v96; // [rsp+60h] [rbp-150h]
  __int64 v97; // [rsp+68h] [rbp-148h]
  _QWORD v98[2]; // [rsp+70h] [rbp-140h] BYREF
  __int64 v99[2]; // [rsp+80h] [rbp-130h] BYREF
  __int64 v100[2]; // [rsp+90h] [rbp-120h] BYREF
  __int64 v101[2]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v102[2]; // [rsp+B0h] [rbp-100h] BYREF
  unsigned __int64 v103; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned int v104; // [rsp+C8h] [rbp-E8h]
  const void *v105; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned int v106; // [rsp+D8h] [rbp-D8h]
  __m128i v107; // [rsp+E0h] [rbp-D0h] BYREF
  const void *v108; // [rsp+F0h] [rbp-C0h] BYREF
  unsigned int v109; // [rsp+F8h] [rbp-B8h]
  __m128i v110; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v111[4]; // [rsp+110h] [rbp-A0h] BYREF
  void *v112; // [rsp+130h] [rbp-80h] BYREF
  unsigned int v113; // [rsp+138h] [rbp-78h]
  unsigned __int64 v114; // [rsp+140h] [rbp-70h] BYREF
  unsigned int v115; // [rsp+148h] [rbp-68h]
  const void *v116; // [rsp+150h] [rbp-60h] BYREF
  unsigned int v117; // [rsp+158h] [rbp-58h]
  unsigned __int64 v118; // [rsp+160h] [rbp-50h] BYREF
  unsigned int v119; // [rsp+168h] [rbp-48h]
  const void *v120; // [rsp+170h] [rbp-40h] BYREF
  unsigned int v121; // [rsp+178h] [rbp-38h]

  v113 = *(_DWORD *)(a1 + 96);
  v112 = &unk_4A16D38;
  sub_AADB10((__int64)&v114, v113, 0);
  sub_AADB10((__int64)&v118, v113, 1);
  v4 = sub_2509740((_QWORD *)(a1 + 72));
  v7 = sub_250D070((_QWORD *)(a1 + 72));
  v5 = *(_BYTE *)v7;
  v6 = (unsigned __int8 *)v7;
  LOBYTE(v7) = *(_BYTE *)v7 == 85;
  if ( v5 <= 0x28u )
    LODWORD(v7) = (0x1041FFFFFFFuLL >> v5) & 1 | v7;
  if ( (_BYTE)v7 )
  {
    v8 = *(_QWORD *)(a1 + 80);
    LOBYTE(v103) = 0;
    v9.m128i_i64[0] = sub_250D2C0((unsigned __int64)v6, v8);
    v110 = v9;
    v10.m128i_i64[0] = sub_2527850(a2, &v110, a1, &v103, 2u);
    v107 = v10;
    if ( !v10.m128i_i8[8] )
      goto LABEL_31;
    if ( v10.m128i_i64[0] )
    {
      v11 = sub_250D2C0(v10.m128i_u64[0], *(_QWORD *)(a1 + 80));
      v13 = sub_2589400(a2, v11, v12, a1, 0, 0, 1);
      if ( v13 )
      {
        (*(void (__fastcall **)(__m128i *, __int64, __int64, unsigned __int64))(*(_QWORD *)v13 + 112LL))(
          &v110,
          v13,
          a2,
          v4);
        sub_254F7F0((__int64)&v112, (__int64)&v110);
        sub_969240(v111);
        sub_969240(v110.m128i_i64);
        if ( v113 )
        {
          if ( !sub_AAF760((__int64)&v114) )
            goto LABEL_31;
        }
      }
    }
    goto LABEL_8;
  }
  v110.m128i_i64[0] = (__int64)v111;
  v110.m128i_i64[1] = 0x400000000LL;
  if ( (unsigned int)v5 - 42 <= 0x11 )
  {
    v20 = *((_QWORD *)v6 - 8);
    v21 = *((_QWORD *)v6 - 4);
    LOBYTE(v98[0]) = 0;
    v88 = v21;
    v22.m128i_i64[0] = sub_250D2C0(v20, *(_QWORD *)(a1 + 80));
    v107 = v22;
    v23 = sub_2527850(a2, &v107, a1, v98, 2u);
    v94 = v23;
    v95 = v24;
    if ( (_BYTE)v24 )
    {
      v81 = v23;
      if ( !v23 )
        goto LABEL_79;
      v25.m128i_i64[0] = sub_250D2C0(v88, *(_QWORD *)(a1 + 80));
      v107 = v25;
      v26 = sub_2527850(a2, &v107, a1, v98, 2u);
      v99[0] = v26;
      v99[1] = v27;
      if ( (_BYTE)v27 )
      {
        if ( !v26 )
          goto LABEL_79;
        if ( *(_BYTE *)(*(_QWORD *)(v81 + 8) + 8LL) != 12 )
          goto LABEL_79;
        if ( *(_BYTE *)(*(_QWORD *)(v26 + 8) + 8LL) != 12 )
          goto LABEL_79;
        v84 = v26;
        v28 = sub_250D2C0(v81, *(_QWORD *)(a1 + 80));
        v30 = sub_2589400(a2, v28, v29, a1, 0, 0, 1);
        if ( !v30 )
          goto LABEL_79;
        v82 = v30;
        sub_255B620((__int64)&v110, v30, v31, v32, v76, v80);
        (*(void (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64))(*(_QWORD *)v82 + 112LL))(
          v101,
          v82,
          a2,
          v4);
        v33 = sub_250D2C0(v84, *(_QWORD *)(a1 + 80));
        v35 = sub_2589400(a2, v33, v34, a1, 0, 0, 1);
        if ( !v35 )
        {
          sub_969240(v102);
          sub_969240(v101);
          goto LABEL_79;
        }
        v85 = v35;
        sub_255B620((__int64)&v110, v35, v36, v77, v37, v38);
        (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64, unsigned __int64))(*(_QWORD *)v85 + 112LL))(
          &v103,
          v85,
          a2,
          v4);
        sub_ABCAA0((__int64)&v107, (__int64)v101, *v6 - 29, (__int64 *)&v103);
        sub_254F7F0((__int64)&v112, (__int64)&v107);
        v39 = sub_2535A50((__int64)&v112);
        sub_969240((__int64 *)&v108);
        sub_969240(v107.m128i_i64);
        sub_969240((__int64 *)&v105);
        sub_969240((__int64 *)&v103);
        sub_969240(v102);
        sub_969240(v101);
        if ( !v39 )
          goto LABEL_79;
      }
    }
    goto LABEL_44;
  }
  if ( (unsigned __int8)(v5 - 82) <= 1u )
  {
    v41 = *((_QWORD *)v6 - 8);
    v42 = *((_QWORD *)v6 - 4);
    v93 = 0;
    v89 = v42;
    v43.m128i_i64[0] = sub_250D2C0(v41, *(_QWORD *)(a1 + 80));
    v107 = v43;
    v44 = sub_2527850(a2, &v107, a1, &v93, 2u);
    v96 = v44;
    v97 = v45;
    if ( !(_BYTE)v45 )
      goto LABEL_44;
    v83 = v44;
    if ( !v44 )
      goto LABEL_79;
    v46.m128i_i64[0] = sub_250D2C0(v89, *(_QWORD *)(a1 + 80));
    v107 = v46;
    v47 = sub_2527850(a2, &v107, a1, &v93, 2u);
    v98[0] = v47;
    v98[1] = v48;
    if ( !(_BYTE)v48 )
      goto LABEL_44;
    if ( !v47 )
      goto LABEL_79;
    if ( *(_BYTE *)(*(_QWORD *)(v83 + 8) + 8LL) != 12 )
      goto LABEL_79;
    if ( *(_BYTE *)(*(_QWORD *)(v47 + 8) + 8LL) != 12 )
      goto LABEL_79;
    v86 = v47;
    v49 = sub_250D2C0(v83, *(_QWORD *)(a1 + 80));
    v54 = sub_2589400(a2, v49, v50, a1, 0, 0, 1);
    if ( !v54 )
      goto LABEL_79;
    v90 = v54;
    sub_255B620((__int64)&v110, v54, v80, v51, v52, v53);
    v55 = sub_250D2C0(v86, *(_QWORD *)(a1 + 80));
    v57 = sub_2589400(a2, v55, v56, a1, 0, 0, 1);
    if ( !v57 )
      goto LABEL_79;
    v87 = v57;
    sub_255B620((__int64)&v110, v57, v58, v59, v60, v78);
    (*(void (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64))(*(_QWORD *)v90 + 112LL))(v99, v90, a2, v4);
    (*(void (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64))(*(_QWORD *)v87 + 112LL))(v101, v87, a2, v4);
    if ( sub_AAF7D0((__int64)v99) || sub_AAF7D0((__int64)v101) )
    {
      sub_969240(v102);
      sub_969240(v101);
      sub_969240(v100);
      sub_969240(v99);
      goto LABEL_44;
    }
    sub_AB15A0((__int64)&v103, *((_WORD *)v6 + 1) & 0x3F, (__int64)v101);
    sub_AB2160((__int64)&v107, (__int64)&v103, (__int64)v99, 0);
    v61 = sub_AAF7D0((__int64)&v107);
    sub_969240((__int64 *)&v108);
    sub_969240(v107.m128i_i64);
    if ( v61 )
    {
      if ( !sub_ABB410(v99, *((_WORD *)v6 + 1) & 0x3F, v101) )
      {
        LODWORD(v95) = 1;
        v94 = 0;
LABEL_77:
        sub_AADBC0((__int64)&v107, (__int64 *)&v94);
        sub_254F7F0((__int64)&v112, (__int64)&v107);
        sub_969240((__int64 *)&v108);
        sub_969240(v107.m128i_i64);
        sub_969240((__int64 *)&v94);
        goto LABEL_78;
      }
    }
    else if ( !sub_ABB410(v99, *((_WORD *)v6 + 1) & 0x3F, v101) )
    {
      sub_AADB10((__int64)&v107, 1u, 1);
      sub_254F7F0((__int64)&v112, (__int64)&v107);
      sub_969240((__int64 *)&v108);
      sub_969240(v107.m128i_i64);
LABEL_78:
      v62 = sub_2535A50((__int64)&v112);
      sub_969240((__int64 *)&v105);
      sub_969240((__int64 *)&v103);
      sub_969240(v102);
      sub_969240(v101);
      sub_969240(v100);
      sub_969240(v99);
      if ( !v62 )
        goto LABEL_79;
      goto LABEL_44;
    }
    LODWORD(v95) = 1;
    v94 = 1;
    goto LABEL_77;
  }
  if ( (unsigned int)v5 - 67 > 0xC )
  {
    sub_2539D60((__int64)&v112);
    goto LABEL_79;
  }
  v65 = *((_QWORD *)v6 - 4);
  v66 = *(_QWORD *)(a1 + 80);
  LOBYTE(v99[0]) = 0;
  v67.m128i_i64[0] = sub_250D2C0(v65, v66);
  v107 = v67;
  v68 = sub_2527850(a2, &v107, a1, v99, 2u);
  v101[0] = v68;
  v101[1] = v69;
  if ( (_BYTE)v69 )
  {
    if ( !v68 )
      goto LABEL_79;
    if ( *(_BYTE *)(*(_QWORD *)(v68 + 8) + 8LL) != 12 )
      goto LABEL_79;
    v70 = sub_250D2C0(v68, *(_QWORD *)(a1 + 80));
    v72 = sub_2589400(a2, v70, v71, a1, 0, 0, 1);
    v75 = v72;
    if ( !v72 )
      goto LABEL_79;
    sub_255B620((__int64)&v110, v72, v79, v80, v73, v74);
    v107.m128i_i32[2] = *(_DWORD *)(v75 + 112);
    if ( v107.m128i_i32[2] > 0x40u )
      sub_C43780((__int64)&v107, (const void **)(v75 + 104));
    else
      v107.m128i_i64[0] = *(_QWORD *)(v75 + 104);
    v109 = *(_DWORD *)(v75 + 128);
    if ( v109 > 0x40 )
      sub_C43780((__int64)&v108, (const void **)(v75 + 120));
    else
      v108 = *(const void **)(v75 + 120);
    sub_AB49F0((__int64)&v103, (__int64)&v107, *v6 - 29, *(_DWORD *)(a1 + 96));
    sub_254F7F0((__int64)&v112, (__int64)&v103);
    sub_969240((__int64 *)&v105);
    sub_969240((__int64 *)&v103);
    sub_969240((__int64 *)&v108);
    sub_969240(v107.m128i_i64);
    if ( !(unsigned __int8)sub_2535A50((__int64)&v112) )
    {
LABEL_79:
      if ( (__int64 *)v110.m128i_i64[0] != v111 )
        _libc_free(v110.m128i_u64[0]);
      goto LABEL_8;
    }
  }
LABEL_44:
  v40 = (_QWORD *)v110.m128i_i64[0];
  v91 = v110.m128i_i64[0] + 8LL * v110.m128i_u32[2];
  if ( v110.m128i_i64[0] != v91 )
  {
    do
    {
      if ( a1 == *v40 )
      {
        v107.m128i_i32[2] = *(_DWORD *)(a1 + 112);
        if ( v107.m128i_i32[2] > 0x40u )
          sub_C43780((__int64)&v107, (const void **)(a1 + 104));
        else
          v107.m128i_i64[0] = *(_QWORD *)(a1 + 104);
        v109 = *(_DWORD *)(a1 + 128);
        if ( v109 > 0x40 )
          sub_C43780((__int64)&v108, (const void **)(a1 + 120));
        else
          v108 = *(const void **)(a1 + 120);
        v104 = v115;
        if ( v115 > 0x40 )
          sub_C43780((__int64)&v103, (const void **)&v114);
        else
          v103 = v114;
        v106 = v117;
        if ( v117 > 0x40 )
          sub_C43780((__int64)&v105, &v116);
        else
          v105 = v116;
        if ( v104 <= 0x40 )
        {
          if ( v103 != v107.m128i_i64[0] )
            goto LABEL_82;
        }
        else if ( !sub_C43C50((__int64)&v103, (const void **)&v107) )
        {
          goto LABEL_82;
        }
        if ( v106 <= 0x40 )
        {
          if ( v105 != v108 )
          {
LABEL_82:
            sub_969240((__int64 *)&v105);
            sub_969240((__int64 *)&v103);
            sub_969240((__int64 *)&v108);
            sub_969240(v107.m128i_i64);
            if ( v115 <= 0x40 && v119 <= 0x40 )
            {
              v115 = v119;
              v114 = v118;
            }
            else
            {
              sub_C43990((__int64)&v114, (__int64)&v118);
            }
            if ( v117 <= 0x40 && v121 <= 0x40 )
            {
              v117 = v121;
              v116 = v120;
            }
            else
            {
              sub_C43990((__int64)&v116, (__int64)&v120);
            }
            goto LABEL_46;
          }
        }
        else if ( !sub_C43C50((__int64)&v105, &v108) )
        {
          goto LABEL_82;
        }
        sub_969240((__int64 *)&v105);
        sub_969240((__int64 *)&v103);
        sub_969240((__int64 *)&v108);
        sub_969240(v107.m128i_i64);
      }
LABEL_46:
      ++v40;
    }
    while ( (_QWORD *)v91 != v40 );
  }
  if ( !v113 )
    goto LABEL_79;
  v18 = sub_AAF760((__int64)&v114);
  if ( (__int64 *)v110.m128i_i64[0] != v111 )
  {
    v92 = v18;
    _libc_free(v110.m128i_u64[0]);
    v18 = v92;
  }
  if ( !v18 )
  {
LABEL_31:
    v16 = sub_255B670(a1 + 88, (__int64)&v112);
    if ( v16 )
      goto LABEL_14;
    v16 = 0;
    v19 = *(_DWORD *)(a1 + 168) + 1;
    *(_DWORD *)(a1 + 168) = v19;
    if ( v19 <= 5 )
      goto LABEL_14;
  }
LABEL_8:
  if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v14 = *(_DWORD *)(a1 + 144), v14 <= 0x40) )
  {
    v64 = *(_QWORD *)(a1 + 136);
    *(_DWORD *)(a1 + 112) = v14;
    *(_QWORD *)(a1 + 104) = v64;
  }
  else
  {
    sub_C43990(a1 + 104, a1 + 136);
  }
  if ( *(_DWORD *)(a1 + 128) <= 0x40u && (v15 = *(_DWORD *)(a1 + 160), v15 <= 0x40) )
  {
    v63 = *(_QWORD *)(a1 + 152);
    *(_DWORD *)(a1 + 128) = v15;
    v16 = 0;
    *(_QWORD *)(a1 + 120) = v63;
  }
  else
  {
    sub_C43990(a1 + 120, a1 + 152);
    v16 = 0;
  }
LABEL_14:
  v112 = &unk_4A16D38;
  if ( v121 > 0x40 && v120 )
    j_j___libc_free_0_0((unsigned __int64)v120);
  if ( v119 > 0x40 && v118 )
    j_j___libc_free_0_0(v118);
  if ( v117 > 0x40 && v116 )
    j_j___libc_free_0_0((unsigned __int64)v116);
  if ( v115 > 0x40 && v114 )
    j_j___libc_free_0_0(v114);
  return v16;
}
