// Function: sub_1B54A50
// Address: 0x1b54a50
//
__int64 __fastcall sub_1B54A50(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // rcx
  unsigned int v14; // r15d
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // dl
  char v29; // dl
  __int64 v30; // r15
  __int64 v31; // r13
  __int64 v32; // r12
  __int64 v33; // r14
  __int64 v34; // rax
  __int64 v35; // rbx
  bool v36; // r15
  __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // r8
  unsigned int v40; // esi
  __int64 v41; // rax
  char v42; // dl
  __int64 v43; // rdx
  char v44; // al
  __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 v47; // r14
  __int64 v48; // r12
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // r8
  _QWORD *v52; // rax
  __int64 v53; // r8
  __int64 *v54; // rdi
  __int64 v55; // rbx
  _QWORD *v56; // r12
  __int64 v57; // r12
  __int64 v58; // r8
  _QWORD *v59; // r15
  _QWORD *i; // r13
  unsigned int v61; // ecx
  __int64 *v62; // rdx
  __int64 v63; // r9
  __int64 v64; // r13
  __int64 v65; // rdx
  __int64 v66; // rcx
  double v67; // xmm4_8
  double v68; // xmm5_8
  __int64 v69; // rax
  __int64 v70; // rax
  int v71; // edx
  int v72; // r10d
  __int64 v73; // rdx
  unsigned __int64 v74; // r13
  __int64 v75; // r12
  int v76; // eax
  unsigned int v77; // r12d
  int v78; // ebx
  bool v79; // [rsp+Fh] [rbp-E1h]
  __int64 v80; // [rsp+20h] [rbp-D0h]
  __int64 v81; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v82; // [rsp+28h] [rbp-C8h]
  __int64 v83; // [rsp+30h] [rbp-C0h]
  __int64 v84; // [rsp+30h] [rbp-C0h]
  __int64 v85; // [rsp+30h] [rbp-C0h]
  __int64 v86; // [rsp+30h] [rbp-C0h]
  __int64 v87; // [rsp+30h] [rbp-C0h]
  __int64 v88; // [rsp+38h] [rbp-B8h]
  __int64 v89; // [rsp+40h] [rbp-B0h]
  __int64 v90; // [rsp+40h] [rbp-B0h]
  __int64 v91; // [rsp+48h] [rbp-A8h]
  __int64 v92; // [rsp+48h] [rbp-A8h]
  int v93; // [rsp+48h] [rbp-A8h]
  __int64 v94; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v95; // [rsp+48h] [rbp-A8h]
  __int64 v96; // [rsp+50h] [rbp-A0h]
  __int64 v97; // [rsp+50h] [rbp-A0h]
  __int64 v98; // [rsp+50h] [rbp-A0h]
  __int64 v99; // [rsp+50h] [rbp-A0h]
  __int64 v100; // [rsp+58h] [rbp-98h]
  _QWORD v101[2]; // [rsp+60h] [rbp-90h] BYREF
  const char *v102; // [rsp+70h] [rbp-80h] BYREF
  __int64 v103; // [rsp+78h] [rbp-78h]
  __int64 v104; // [rsp+80h] [rbp-70h]
  unsigned int v105; // [rsp+88h] [rbp-68h]
  __m128i v106; // [rsp+90h] [rbp-60h] BYREF
  __int64 v107; // [rsp+A0h] [rbp-50h]
  __int64 v108; // [rsp+A8h] [rbp-48h]
  __int64 v109; // [rsp+B0h] [rbp-40h]

  v13 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v13 + 16) != 77 )
    return 0;
  v16 = *(_QWORD *)(a1 + 40);
  v17 = a1;
  v18 = *(_QWORD *)(v13 + 40);
  v100 = v16;
  if ( v18 != v16 )
    return 0;
  v19 = *(_QWORD *)(v13 + 8);
  if ( !v19 || *(_QWORD *)(v19 + 8) )
    return 0;
  if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) == 1 )
  {
    v14 = 1;
    sub_1AA62D0(v18, 0, a5, a6, a7, a8, a9, a10, a11, a12);
    return v14;
  }
  v91 = v13;
  if ( byte_4FB6F80 || !(unsigned __int8)sub_1B45E10((_BYTE *)v100) )
    return 0;
  v22 = v100 + 40;
  v23 = v91;
  if ( v100 + 40 == *(_QWORD *)(v100 + 48) )
    goto LABEL_24;
  v92 = v17;
  v24 = *(_QWORD *)(v100 + 48);
  v25 = v23;
  do
  {
    if ( !v24 )
LABEL_99:
      BUG();
    if ( *(_BYTE *)(v24 - 8) != 78 )
      goto LABEL_19;
    if ( (unsigned __int8)sub_1560260((_QWORD *)(v24 + 32), -1, 24)
      || (v26 = *(_QWORD *)(v24 - 48), !*(_BYTE *)(v26 + 16))
      && (v106.m128i_i64[0] = *(_QWORD *)(v26 + 112), (unsigned __int8)sub_1560260(&v106, -1, 24))
      || (unsigned __int8)sub_1560260((_QWORD *)(v24 + 32), -1, 8) )
    {
LABEL_23:
      v23 = v25;
      v30 = v24;
      v17 = v92;
      if ( v22 == v30 )
        goto LABEL_24;
      return 0;
    }
    v27 = *(_QWORD *)(v24 - 48);
    v28 = *(_BYTE *)(v27 + 16);
    if ( v28 )
    {
      if ( v28 != 20 )
        goto LABEL_19;
LABEL_18:
      if ( *(_BYTE *)(v27 + 96) )
        goto LABEL_23;
      goto LABEL_19;
    }
    v106.m128i_i64[0] = *(_QWORD *)(v27 + 112);
    if ( (unsigned __int8)sub_1560260(&v106, -1, 8) )
      goto LABEL_23;
    v27 = *(_QWORD *)(v24 - 48);
    v29 = *(_BYTE *)(v27 + 16);
    if ( v29 == 20 )
      goto LABEL_18;
    if ( !v29 )
    {
      v84 = v27 + 112;
      v45 = v27 + 112;
      if ( !(unsigned __int8)sub_1560180(v27 + 112, 36)
        && !(unsigned __int8)sub_1560180(v45, 36)
        && !(unsigned __int8)sub_1560180(v84, 37) )
      {
        goto LABEL_23;
      }
    }
LABEL_19:
    v24 = *(_QWORD *)(v24 + 8);
  }
  while ( v22 != v24 );
  v17 = v92;
  v23 = v25;
LABEL_24:
  v93 = *(_DWORD *)(v23 + 20) & 0xFFFFFFF;
  if ( !v93 )
    return 0;
  v88 = v17;
  v82 = a2;
  v31 = v100 + 40;
  v80 = a4;
  v32 = 0;
  v96 = a3;
  v33 = v23;
  while ( 1 )
  {
    v34 = sub_1455F60(v33, v32);
    v35 = v34;
    if ( *(_BYTE *)(v34 + 16) == 13 )
    {
      v36 = sub_1642F90(*(_QWORD *)v34, 1);
      if ( v36 )
      {
        v37 = *(_QWORD *)(sub_193FF80(v33) + 8 * v32);
        v38 = *(_QWORD **)(v35 + 24);
        if ( *(_DWORD *)(v35 + 32) > 0x40u )
          v38 = (_QWORD *)*v38;
        v39 = *(_QWORD *)(v88 + -24 - 24LL * (v38 == 0));
        if ( !v39 || v39 != v100 )
          break;
      }
    }
LABEL_46:
    if ( v93 == (_DWORD)++v32 )
      return 0;
  }
  v40 = 0;
  v41 = *(_QWORD *)(v100 + 48);
  if ( v31 == v41 )
    goto LABEL_53;
  while ( 2 )
  {
    if ( !v41 )
      BUG();
    v42 = *(_BYTE *)(v41 - 8);
    if ( v42 != 78 )
    {
      if ( v42 == 77 && v33 == v41 - 24 )
        goto LABEL_37;
LABEL_43:
      ++v40;
      goto LABEL_37;
    }
    v43 = *(_QWORD *)(v41 - 48);
    if ( *(_BYTE *)(v43 + 16) || (*(_BYTE *)(v43 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v43 + 36) - 35) > 3 )
      goto LABEL_43;
LABEL_37:
    v41 = *(_QWORD *)(v41 + 8);
    if ( v31 != v41 )
      continue;
    break;
  }
  if ( v40 > 1 )
  {
    v83 = v39;
    v89 = v37;
    v44 = sub_1C07420(v96, v37, v100);
    v37 = v89;
    v39 = v83;
    if ( !v44 )
      goto LABEL_46;
  }
LABEL_53:
  v85 = v39;
  v90 = v37;
  if ( *(_BYTE *)(sub_157EBA0(v37) + 16) == 28 )
    goto LABEL_46;
  v46 = v85;
  v47 = v96;
  v97 = v85;
  v86 = *(_QWORD *)(v85 + 56);
  v48 = v80;
  v102 = sub_1649960(v46);
  v103 = v49;
  v106.m128i_i64[0] = (__int64)&v102;
  LOWORD(v107) = 773;
  v106.m128i_i64[1] = (__int64)".critedge";
  v94 = sub_157E9C0(v100);
  v50 = (_QWORD *)sub_22077B0(64);
  v51 = v97;
  v81 = (__int64)v50;
  if ( v50 )
  {
    sub_157FB60(v50, v94, (__int64)&v106, v86, v97);
    v51 = v97;
  }
  v98 = v51;
  v52 = sub_1648A60(56, 1u);
  v53 = v98;
  if ( v52 )
  {
    sub_15F8590((__int64)v52, v98, v81);
    v53 = v98;
  }
  sub_1B44430(v53, v81, v100);
  v102 = 0;
  v103 = 0;
  v54 = *(__int64 **)(v81 + 48);
  v104 = 0;
  v105 = 0;
  v55 = *(_QWORD *)(v100 + 48);
  v79 = v36;
  v87 = v48;
  while ( 2 )
  {
    if ( !v55 )
      goto LABEL_99;
    if ( v88 != v55 - 24 )
    {
      if ( *(_BYTE *)(v55 - 8) == 77 )
      {
        v106.m128i_i64[0] = v55 - 24;
        v56 = sub_176FB00((__int64)&v102, v106.m128i_i64);
        v56[1] = sub_1455EB0(v55 - 24, v90);
      }
      else
      {
        v57 = sub_15F4880(v55 - 24);
        if ( (*(_BYTE *)(v55 - 1) & 0x20) != 0 )
        {
          v101[0] = sub_1649960(v55 - 24);
          LOWORD(v107) = 773;
          v106.m128i_i64[0] = (__int64)v101;
          v106.m128i_i64[1] = (__int64)".c";
          v101[1] = v73;
          sub_164B780(v57, v106.m128i_i64);
        }
        v59 = (_QWORD *)sub_13CF970(v57);
        for ( i = &v59[3 * (*(_DWORD *)(v57 + 20) & 0xFFFFFFF)]; v59 != i; v59 += 3 )
        {
          if ( v105 )
          {
            v58 = v105 - 1;
            v61 = v58 & (((unsigned int)*v59 >> 9) ^ ((unsigned int)*v59 >> 4));
            v62 = (__int64 *)(v103 + 16LL * v61);
            v63 = *v62;
            if ( *v59 == *v62 )
            {
LABEL_69:
              if ( v62 != (__int64 *)(v103 + 16LL * v105) )
                sub_1593B40(v59, v62[1]);
            }
            else
            {
              v71 = 1;
              while ( v63 != -8 )
              {
                v72 = v71 + 1;
                v61 = v58 & (v71 + v61);
                v62 = (__int64 *)(v103 + 16LL * v61);
                v63 = *v62;
                if ( *v59 == *v62 )
                  goto LABEL_69;
                v71 = v72;
              }
            }
          }
        }
        v106 = (__m128i)v82;
        v107 = 0;
        v108 = v87;
        v109 = 0;
        v64 = sub_13E3350(v57, &v106, 0, 1, v58);
        if ( v64 )
        {
          if ( *(_QWORD *)(v55 - 16) )
          {
            v106.m128i_i64[0] = v55 - 24;
            sub_176FB00((__int64)&v102, v106.m128i_i64)[1] = v64;
          }
          if ( sub_1B46960(v57) )
            goto LABEL_76;
          sub_164BEC0(v57, (__int64)&v106, v65, v66, a5, a6, a7, a8, v67, v68, a11, a12);
        }
        else
        {
          if ( *(_QWORD *)(v55 - 16) )
          {
            v106.m128i_i64[0] = v55 - 24;
            sub_176FB00((__int64)&v102, v106.m128i_i64)[1] = v57;
          }
LABEL_76:
          sub_157E9D0(v81 + 40, v57);
          v69 = *v54;
          *(_QWORD *)(v57 + 32) = v54;
          *(_QWORD *)(v57 + 24) = v69 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(v57 + 24) & 7LL;
          *(_QWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v57 + 24;
          *v54 = *v54 & 7 | (v57 + 24);
          if ( *(_BYTE *)(v57 + 16) == 78 )
          {
            v70 = *(_QWORD *)(v57 - 24);
            if ( !*(_BYTE *)(v70 + 16) && (*(_BYTE *)(v70 + 33) & 0x20) != 0 && *(_DWORD *)(v70 + 36) == 4 )
              sub_14CE830(v87, v57);
          }
        }
      }
      v55 = *(_QWORD *)(v55 + 8);
      continue;
    }
    break;
  }
  v99 = v55 - 24;
  v14 = v79;
  v74 = v82;
  v75 = v87;
  v95 = sub_157EBA0(v90);
  v76 = sub_15F4D60(v95);
  if ( v76 )
  {
    v77 = 0;
    v78 = v76;
    do
    {
      if ( v100 == sub_15F4DF0(v95, v77) )
      {
        sub_157F2D0(v100, v90, 0);
        sub_15F4ED0(v95, v77, v81);
      }
      ++v77;
    }
    while ( v78 != v77 );
    v14 = v79;
    v74 = v82;
    v75 = v87;
  }
  sub_1B54A50(v99, v74, v47, v75);
  j___libc_free_0(v103);
  return v14;
}
