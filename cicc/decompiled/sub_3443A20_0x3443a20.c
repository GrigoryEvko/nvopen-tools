// Function: sub_3443A20
// Address: 0x3443a20
//
__int64 __fastcall sub_3443A20(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r13
  unsigned int v4; // r12d
  bool v5; // r14
  __int64 result; // rax
  __int64 v7; // rbx
  char v8; // si
  __int64 v9; // rax
  char v10; // cl
  unsigned int v11; // r12d
  bool v12; // dl
  unsigned int v13; // r12d
  bool v14; // dl
  unsigned int v15; // r13d
  unsigned int v16; // edx
  int v17; // eax
  __int64 v18; // r12
  unsigned int v19; // r14d
  unsigned int v20; // r13d
  bool v21; // dl
  unsigned int v22; // r13d
  unsigned __int64 v23; // rax
  char v24; // cl
  __int64 v25; // rdx
  unsigned int v26; // edx
  char v27; // cl
  unsigned int v28; // esi
  unsigned __int64 v29; // rdx
  unsigned int v30; // eax
  unsigned int v31; // ecx
  unsigned int v32; // eax
  __int64 v33; // rax
  unsigned int v34; // r13d
  bool v35; // al
  __int64 v36; // r13
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int8 *v39; // r14
  __int64 v40; // rax
  unsigned __int8 *v41; // rdx
  unsigned __int8 *v42; // r15
  unsigned __int8 **v43; // rax
  __int64 v44; // r13
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned __int8 *v47; // r14
  __int64 v48; // rax
  unsigned __int8 *v49; // rdx
  unsigned __int8 *v50; // r15
  unsigned __int8 **v51; // rax
  __int64 v52; // r13
  __int64 v53; // r15
  __int64 v54; // r14
  unsigned __int16 v55; // ax
  _QWORD *v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // eax
  unsigned __int64 v59; // r12
  unsigned __int8 *v60; // rax
  unsigned __int8 *v61; // rdx
  unsigned __int8 *v62; // r13
  __int64 v63; // rdx
  unsigned __int8 *v64; // r12
  unsigned __int8 **v65; // rdx
  __int64 v66; // r12
  unsigned __int8 *v67; // rax
  __int64 v68; // r9
  unsigned __int8 *v69; // rdx
  unsigned __int8 *v70; // r15
  __int64 v71; // rdx
  unsigned __int8 *v72; // r14
  unsigned __int8 **v73; // rdx
  bool v74; // dl
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rdx
  unsigned int v77; // r12d
  unsigned int v78; // eax
  unsigned __int64 v79; // rax
  unsigned __int64 v82; // rdx
  _QWORD *v83; // rdx
  __int64 v84; // [rsp-8h] [rbp-F8h]
  __int64 v85; // [rsp+8h] [rbp-E8h]
  unsigned int v86; // [rsp+20h] [rbp-D0h]
  unsigned int v87; // [rsp+24h] [rbp-CCh]
  __int64 v88; // [rsp+30h] [rbp-C0h]
  unsigned int v89; // [rsp+38h] [rbp-B8h]
  int v90; // [rsp+38h] [rbp-B8h]
  __int64 v91; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v92; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v93; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v94; // [rsp+58h] [rbp-98h]
  _QWORD *v95; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v96; // [rsp+68h] [rbp-88h]
  unsigned __int64 v97; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v98; // [rsp+78h] [rbp-78h]
  _QWORD *v99; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v100; // [rsp+88h] [rbp-68h]
  unsigned __int64 v101; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v102; // [rsp+98h] [rbp-58h]
  _QWORD *v103; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v104; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v105; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v106; // [rsp+B8h] [rbp-38h]

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
    v5 = *(_QWORD *)(v3 + 24) == 0;
  else
    v5 = v4 == (unsigned int)sub_C444A0(v3 + 24);
  result = 0;
  if ( v5 )
    return result;
  v92 = v4;
  v7 = *a1;
  if ( v4 <= 0x40 )
  {
    v91 = *(_QWORD *)(v3 + 24);
    v8 = (v4 - 1) & 0x3F;
    v9 = 1LL << ((unsigned __int8)v4 - 1);
LABEL_6:
    v10 = v92;
    if ( (v9 & v91) == 0 )
    {
LABEL_7:
      v5 = 1LL << (v10 - 1) == v91;
      goto LABEL_8;
    }
    v76 = (0xFFFFFFFFFFFFFFFFLL >> (63 - v8)) & ~v91;
    if ( !v4 )
      v76 = 0;
    v91 = v76;
    goto LABEL_114;
  }
  sub_C43780((__int64)&v91, (const void **)(v3 + 24));
  v4 = v92;
  v9 = 1LL << ((unsigned __int8)v92 - 1);
  v8 = (v92 - 1) & 0x3F;
  if ( v92 <= 0x40 )
    goto LABEL_6;
  if ( (*(_QWORD *)(v91 + 8LL * ((v92 - 1) >> 6)) & v9) == 0 )
    goto LABEL_8;
  sub_C43D10((__int64)&v91);
LABEL_114:
  sub_C46250((__int64)&v91);
  v10 = v92;
  if ( v92 <= 0x40 )
    goto LABEL_7;
  if ( (*(_QWORD *)(v91 + 8LL * ((v92 - 1) >> 6)) & (1LL << ((unsigned __int8)v92 - 1))) != 0 )
  {
    v77 = v92 - 1;
    v5 = (unsigned int)sub_C44590((__int64)&v91) == v77;
  }
LABEL_8:
  **(_BYTE **)v7 |= v5;
  v11 = v92;
  if ( v92 <= 0x40 )
    v12 = v91 == 1;
  else
    v12 = v11 - 1 == (unsigned int)sub_C444A0((__int64)&v91);
  **(_BYTE **)(v7 + 8) |= v12;
  v13 = v92;
  if ( v92 <= 0x40 )
    v14 = v91 == 1;
  else
    v14 = v13 - 1 == (unsigned int)sub_C444A0((__int64)&v91);
  **(_BYTE **)(v7 + 16) &= v14;
  v15 = v92;
  v16 = v92;
  if ( v92 <= 0x40 )
  {
    _RAX = v91;
    LODWORD(v18) = 64;
    v94 = v92;
    __asm { tzcnt   rcx, rax }
    v93 = v91;
    if ( v91 )
      LODWORD(v18) = _RCX;
    if ( v92 <= (unsigned int)v18 )
      LODWORD(v18) = v92;
  }
  else
  {
    v17 = sub_C44590((__int64)&v91);
    v94 = v15;
    LODWORD(v18) = v17;
    sub_C43780((__int64)&v93, (const void **)&v91);
    v16 = v94;
    if ( v94 > 0x40 )
    {
      sub_C482E0((__int64)&v93, v18);
      v15 = v92;
      goto LABEL_15;
    }
    v15 = v92;
  }
  if ( v16 == (_DWORD)v18 )
    v93 = 0;
  else
    v93 >>= v18;
LABEL_15:
  v19 = v15 - 1;
  if ( v15 > 0x40 )
  {
    if ( (*(_QWORD *)(v91 + 8LL * (v19 >> 6)) & (1LL << v19)) != 0 && (unsigned int)sub_C44590((__int64)&v91) == v19 )
      goto LABEL_18;
    goto LABEL_17;
  }
  if ( v91 != 1LL << v19 )
LABEL_17:
    **(_BYTE **)(v7 + 24) |= (_DWORD)v18 != 0;
LABEL_18:
  v20 = v94;
  if ( v94 <= 0x40 )
    v21 = v93 == 1;
  else
    v21 = v20 - 1 == (unsigned int)sub_C444A0((__int64)&v93);
  **(_BYTE **)(v7 + 32) &= v21;
  v22 = v92;
  sub_C473B0((__int64)&v95, (__int64)&v93);
  LODWORD(v106) = v22;
  if ( v22 > 0x40 )
  {
    sub_C43690((__int64)&v105, -1, 1);
    v87 = v22 - 1;
    v85 = 1LL << ((unsigned __int8)v22 - 1);
    v25 = ~v85;
    if ( (unsigned int)v106 > 0x40 )
    {
      *(_QWORD *)(v105 + 8LL * (v87 >> 6)) &= v25;
      goto LABEL_25;
    }
  }
  else
  {
    v87 = v22 - 1;
    v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
    v24 = v22 - 1;
    if ( !v22 )
      v23 = 0;
    v105 = v23;
    v85 = 1LL << v24;
    v25 = ~(1LL << v24);
  }
  v105 &= v25;
LABEL_25:
  sub_C4A1D0((__int64)&v97, (__int64)&v105, (__int64)&v93);
  if ( (unsigned int)v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  v26 = v98;
  LODWORD(v106) = v98;
  v27 = v18 - v98;
  if ( v98 <= 0x40 )
  {
    v105 = 0;
    v28 = v18;
    if ( v98 == (_DWORD)v18 )
    {
      v30 = v18;
      v29 = 0;
      goto LABEL_35;
    }
    goto LABEL_30;
  }
  v90 = v18 - v98;
  sub_C43690((__int64)&v105, 0, 0);
  v26 = v106;
  v27 = v90;
  v28 = v106 + v90;
  if ( (_DWORD)v106 != (_DWORD)v106 + v90 )
  {
LABEL_30:
    if ( v28 > 0x3F || v26 > 0x40 )
      sub_C43C90(&v105, v28, v26);
    else
      v105 |= 0xFFFFFFFFFFFFFFFFLL >> (v27 + 64) << v28;
  }
  if ( v98 > 0x40 )
  {
    sub_C43B90(&v97, (__int64 *)&v105);
    v30 = v106;
    goto LABEL_36;
  }
  v29 = v105;
  v30 = v106;
LABEL_35:
  v97 &= v29;
LABEL_36:
  if ( v30 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  v31 = v92 - 1;
  if ( v92 <= 0x40 )
  {
    if ( v91 == 1LL << v31 )
      goto LABEL_42;
LABEL_99:
    v89 = v98;
    if ( v98 <= 0x40 )
    {
      v75 = v97;
    }
    else
    {
      if ( v89 - (unsigned int)sub_C444A0((__int64)&v97) > 0x40 )
      {
        v74 = 1;
        goto LABEL_102;
      }
      v75 = *(_QWORD *)v97;
    }
    v74 = v75 != 0;
LABEL_102:
    **(_BYTE **)(v7 + 40) |= v74;
    v104 = v98;
    if ( v98 <= 0x40 )
      goto LABEL_43;
    goto LABEL_103;
  }
  if ( (*(_QWORD *)(v91 + 8LL * (v31 >> 6)) & (1LL << v31)) == 0 || v31 != (unsigned int)sub_C44590((__int64)&v91) )
    goto LABEL_99;
LABEL_42:
  v104 = v98;
  if ( v98 <= 0x40 )
  {
LABEL_43:
    v103 = (_QWORD *)v97;
    goto LABEL_44;
  }
LABEL_103:
  sub_C43780((__int64)&v103, (const void **)&v97);
LABEL_44:
  sub_C47170((__int64)&v103, 2u);
  v32 = v104;
  LODWORD(v106) = v22;
  v104 = 0;
  v102 = v32;
  v101 = (unsigned __int64)v103;
  v33 = 1LL << v18;
  if ( v22 > 0x40 )
  {
    sub_C43690((__int64)&v105, 0, 0);
    v33 = 1LL << v18;
    if ( (unsigned int)v106 > 0x40 )
    {
      *(_QWORD *)(v105 + 8LL * ((unsigned int)v18 >> 6)) |= 1LL << v18;
      goto LABEL_47;
    }
  }
  else
  {
    v105 = 0;
  }
  v105 |= v33;
LABEL_47:
  sub_C4A1D0((__int64)&v99, (__int64)&v101, (__int64)&v105);
  if ( (unsigned int)v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v104 > 0x40 && v103 )
    j_j___libc_free_0_0((unsigned __int64)v103);
  if ( v94 <= 0x40 )
  {
    if ( v93 != 1 )
      goto LABEL_58;
LABEL_122:
    LODWORD(v106) = v22;
    if ( v22 > 0x40 )
    {
      sub_C43690((__int64)&v105, 0, 0);
      if ( (unsigned int)v106 > 0x40 )
      {
        *(_QWORD *)(v105 + 8LL * (v87 >> 6)) |= v85;
LABEL_125:
        if ( v98 > 0x40 && v97 )
          j_j___libc_free_0_0(v97);
        v97 = v105;
        v78 = v106;
        LODWORD(v106) = v22 - v18;
        v98 = v78;
        if ( v22 - (unsigned int)v18 > 0x40 )
        {
          sub_C43690((__int64)&v105, -1, 1);
        }
        else
        {
          v79 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v18 - (unsigned __int8)v22);
          if ( v22 == (_DWORD)v18 )
            v79 = 0;
          v105 = v79;
        }
        sub_C449B0((__int64)&v103, (const void **)&v105, v22);
        if ( v100 > 0x40 && v99 )
          j_j___libc_free_0_0((unsigned __int64)v99);
        v99 = v103;
        v100 = v104;
        if ( (unsigned int)v106 > 0x40 && v105 )
          j_j___libc_free_0_0(v105);
        goto LABEL_58;
      }
    }
    else
    {
      v105 = 0;
    }
    v105 |= v85;
    goto LABEL_125;
  }
  v86 = v94;
  if ( (unsigned int)sub_C444A0((__int64)&v93) == v86 - 1 )
    goto LABEL_122;
LABEL_58:
  v34 = v92;
  if ( v92 <= 0x40 )
    v35 = v91 == 1;
  else
    v35 = v34 - 1 == (unsigned int)sub_C444A0((__int64)&v91);
  v18 = (unsigned int)v18;
  if ( v35 )
  {
    if ( v96 > 0x40 )
    {
      *v95 = 0;
      memset(v95 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v96 + 63) >> 6) - 8);
    }
    else
    {
      v95 = 0;
    }
    if ( v98 > 0x40 )
    {
      *(_QWORD *)v97 = -1;
      memset((void *)(v97 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v98 + 63) >> 6) - 8);
    }
    else
    {
      v82 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v98;
      if ( !v98 )
        v82 = 0;
      v97 = v82;
    }
    if ( v100 > 0x40 )
    {
      v18 = 0xFFFFFFFFLL;
      *v99 = -1;
      memset(v99 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v100 + 63) >> 6) - 8);
    }
    else
    {
      v18 = 0xFFFFFFFFLL;
      v83 = (_QWORD *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v100);
      if ( !v100 )
        v83 = 0;
      v99 = v83;
    }
  }
  v36 = *(_QWORD *)(v7 + 48);
  v39 = sub_34007B0(
          *(_QWORD *)(v7 + 56),
          (__int64)&v95,
          *(_QWORD *)(v7 + 64),
          **(_DWORD **)(v7 + 72),
          *(_QWORD *)(*(_QWORD *)(v7 + 72) + 8LL),
          0,
          a3,
          0);
  v40 = *(unsigned int *)(v36 + 8);
  v42 = v41;
  if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 12) )
  {
    sub_C8D5F0(v36, (const void *)(v36 + 16), v40 + 1, 0x10u, v37, v38);
    v40 = *(unsigned int *)(v36 + 8);
  }
  v43 = (unsigned __int8 **)(*(_QWORD *)v36 + 16 * v40);
  *v43 = v39;
  v43[1] = v42;
  ++*(_DWORD *)(v36 + 8);
  v44 = *(_QWORD *)(v7 + 80);
  v47 = sub_34007B0(
          *(_QWORD *)(v7 + 56),
          (__int64)&v97,
          *(_QWORD *)(v7 + 64),
          **(_DWORD **)(v7 + 72),
          *(_QWORD *)(*(_QWORD *)(v7 + 72) + 8LL),
          0,
          a3,
          0);
  v48 = *(unsigned int *)(v44 + 8);
  v50 = v49;
  if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v44 + 12) )
  {
    sub_C8D5F0(v44, (const void *)(v44 + 16), v48 + 1, 0x10u, v45, v46);
    v48 = *(unsigned int *)(v44 + 8);
  }
  v51 = (unsigned __int8 **)(*(_QWORD *)v44 + 16 * v48);
  *v51 = v47;
  v51[1] = v50;
  ++*(_DWORD *)(v44 + 8);
  v52 = *(_QWORD *)(v7 + 96);
  v53 = *(_QWORD *)(v7 + 88);
  v88 = *(_QWORD *)(v7 + 56);
  v54 = *(_QWORD *)(v7 + 64);
  v55 = *(_WORD *)v52;
  if ( *(_WORD *)v52 )
  {
    if ( v55 == 1 || (unsigned __int16)(v55 - 504) <= 7u )
      BUG();
    v57 = 16LL * (v55 - 1);
    v56 = *(_QWORD **)&byte_444C4A0[v57];
    LOBYTE(v57) = byte_444C4A0[v57 + 8];
  }
  else
  {
    v56 = (_QWORD *)sub_3007260(v52);
    v105 = (unsigned __int64)v56;
    v106 = v57;
  }
  v103 = v56;
  LOBYTE(v104) = v57;
  v58 = sub_CA1930(&v103);
  v102 = v58;
  if ( v58 > 0x40 )
  {
    sub_C43690((__int64)&v101, v18, 0);
  }
  else
  {
    v59 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v58) & v18;
    if ( !v58 )
      v59 = 0;
    v101 = v59;
  }
  v60 = sub_34007B0(v88, (__int64)&v101, v54, *(_DWORD *)v52, *(_QWORD *)(v52 + 8), 0, a3, 0);
  v62 = v61;
  v63 = *(unsigned int *)(v53 + 8);
  v64 = v60;
  if ( v63 + 1 > (unsigned __int64)*(unsigned int *)(v53 + 12) )
  {
    sub_C8D5F0(v53, (const void *)(v53 + 16), v63 + 1, 0x10u, v63 + 1, v84);
    v63 = *(unsigned int *)(v53 + 8);
  }
  v65 = (unsigned __int8 **)(*(_QWORD *)v53 + 16 * v63);
  *v65 = v64;
  v65[1] = v62;
  ++*(_DWORD *)(v53 + 8);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  v66 = *(_QWORD *)(v7 + 104);
  v67 = sub_34007B0(
          *(_QWORD *)(v7 + 56),
          (__int64)&v99,
          *(_QWORD *)(v7 + 64),
          **(_DWORD **)(v7 + 72),
          *(_QWORD *)(*(_QWORD *)(v7 + 72) + 8LL),
          0,
          a3,
          0);
  v70 = v69;
  v71 = *(unsigned int *)(v66 + 8);
  v72 = v67;
  if ( v71 + 1 > (unsigned __int64)*(unsigned int *)(v66 + 12) )
  {
    sub_C8D5F0(v66, (const void *)(v66 + 16), v71 + 1, 0x10u, v71 + 1, v68);
    v71 = *(unsigned int *)(v66 + 8);
  }
  v73 = (unsigned __int8 **)(*(_QWORD *)v66 + 16 * v71);
  *v73 = v72;
  v73[1] = v70;
  ++*(_DWORD *)(v66 + 8);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0((unsigned __int64)v99);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0((unsigned __int64)v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v92 > 0x40 )
  {
    if ( v91 )
      j_j___libc_free_0_0(v91);
  }
  return 1;
}
