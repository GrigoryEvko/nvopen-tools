// Function: sub_178B660
// Address: 0x178b660
//
__int64 __fastcall sub_178B660(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  int v6; // r12d
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // r15
  unsigned int v14; // r13d
  __int64 v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 i; // r15
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned int v25; // eax
  const char *v26; // rax
  __int64 v27; // rsi
  int v28; // r12d
  int v29; // r12d
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r10
  __int64 v37; // rdx
  __int64 v38; // r12
  int v39; // eax
  __int64 v40; // rax
  int v41; // edx
  __int64 v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rcx
  unsigned __int64 v45; // rsi
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rcx
  _QWORD *v49; // rax
  __int64 v50; // r10
  char *v51; // r12
  int v52; // r14d
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // r10
  __int64 v57; // r12
  unsigned int j; // r13d
  __int64 v59; // rdx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // r10
  int v65; // eax
  __int64 v66; // rax
  int v67; // ecx
  __int64 v68; // rcx
  __int64 *v69; // rax
  __int64 v70; // rsi
  unsigned __int64 v71; // rcx
  __int64 v72; // rcx
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // rdx
  unsigned __int64 v76; // rcx
  __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // [rsp+8h] [rbp-B8h]
  __int64 v83; // [rsp+18h] [rbp-A8h]
  __int64 v84; // [rsp+18h] [rbp-A8h]
  __int64 v85; // [rsp+18h] [rbp-A8h]
  int v86; // [rsp+18h] [rbp-A8h]
  __int64 v87; // [rsp+18h] [rbp-A8h]
  int v88; // [rsp+20h] [rbp-A0h]
  __int64 v89; // [rsp+20h] [rbp-A0h]
  char v90; // [rsp+2Ch] [rbp-94h]
  unsigned int v91; // [rsp+30h] [rbp-90h]
  unsigned int v92; // [rsp+30h] [rbp-90h]
  __int64 v93; // [rsp+30h] [rbp-90h]
  __int64 v94; // [rsp+30h] [rbp-90h]
  int v95; // [rsp+38h] [rbp-88h]
  __int64 v96; // [rsp+38h] [rbp-88h]
  __int64 v97; // [rsp+38h] [rbp-88h]
  __int64 v98; // [rsp+38h] [rbp-88h]
  __int64 v99; // [rsp+38h] [rbp-88h]
  __int64 v100; // [rsp+38h] [rbp-88h]
  __int64 *v102; // [rsp+48h] [rbp-78h]
  _QWORD v103[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v104; // [rsp+60h] [rbp-60h] BYREF
  __int64 v105; // [rsp+68h] [rbp-58h]
  __int64 v106; // [rsp+70h] [rbp-50h]
  __int64 v107; // [rsp+78h] [rbp-48h]
  int v108; // [rsp+80h] [rbp-40h]
  char v109; // [rsp+84h] [rbp-3Ch] BYREF

  v2 = a2;
  v102 = (__int64 *)a2;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = *v3;
  if ( sub_15F32D0(*v3) )
    return 0;
  v5 = **(_QWORD **)(v4 - 24);
  if ( *(_BYTE *)(v5 + 8) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v6 = *(_DWORD *)(v5 + 8) >> 8;
  v95 = v6;
  if ( v6 == 5 || !v6 )
    return 0;
  v7 = *(_QWORD *)(v4 + 40);
  v8 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( v7 != *(_QWORD *)(v8 + 24LL * *(unsigned int *)(a2 + 56) + 8) )
    return 0;
  v9 = *(_QWORD *)(v4 + 32);
  v10 = v7 + 40;
  for ( LOWORD(v91) = *(_WORD *)(v4 + 18); v10 != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    v11 = v9 - 24;
    if ( !v9 )
      v11 = 0;
    if ( (unsigned __int8)sub_15F3040(v11) )
      return 0;
  }
  if ( !(unsigned __int8)sub_1789010(v4) )
    return 0;
  v90 = v91 & 1;
  if ( (v91 & 1) != 0 )
  {
    v24 = sub_157EBA0(*(_QWORD *)(v4 + 40));
    if ( (unsigned int)sub_15F4D60(v24) != 1 )
      return 0;
  }
  v92 = 1 << (v91 >> 1) >> 1;
  v88 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v88 != 1 )
  {
    v83 = v4;
    v14 = 1;
    while ( 1 )
    {
      v16 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v17 = *(_QWORD *)(v16 + 24LL * v14);
      if ( *(_BYTE *)(v17 + 16) != 54 )
        return 0;
      v18 = *(_QWORD *)(v17 + 8);
      if ( !v18 )
        return 0;
      if ( *(_QWORD *)(v18 + 8) )
        return 0;
      if ( (*(_BYTE *)(v17 + 18) & 1) != v90 )
        return 0;
      v19 = *(_QWORD *)(v17 + 40);
      if ( *(_QWORD *)(v16 + 8LL * v14 + 24LL * *(unsigned int *)(a2 + 56) + 8) != v19 )
        return 0;
      v20 = **(_QWORD **)(v17 - 24);
      if ( *(_BYTE *)(v20 + 8) == 16 )
        v20 = **(_QWORD **)(v20 + 16);
      if ( v95 != *(_DWORD *)(v20 + 8) >> 8 )
        return 0;
      v21 = *(_QWORD *)(v17 + 32);
      for ( i = v19 + 40; i != v21; v21 = *(_QWORD *)(v21 + 8) )
      {
        v23 = v21 - 24;
        if ( !v21 )
          v23 = 0;
        if ( (unsigned __int8)sub_15F3040(v23) )
          return 0;
      }
      if ( !(unsigned __int8)sub_1789010(v17) )
        return 0;
      v25 = 1 << (*(unsigned __int16 *)(v17 + 18) >> 1) >> 1;
      if ( (v92 != 0) != (v25 != 0) )
        return 0;
      if ( v92 <= v25 )
        v25 = v92;
      v92 = v25;
      if ( v90 )
      {
        v54 = sub_157EBA0(*(_QWORD *)(v17 + 40));
        if ( (unsigned int)sub_15F4D60(v54) != 1 )
          return 0;
      }
      if ( v88 == ++v14 )
      {
        v2 = a2;
        v4 = v83;
        break;
      }
    }
  }
  v26 = sub_1649960(v2);
  v27 = 773;
  v28 = *(_DWORD *)(v2 + 20);
  v103[0] = v26;
  v104 = (__int64)v103;
  v29 = v28 & 0xFFFFFFF;
  v103[1] = v30;
  LOWORD(v106) = 773;
  v105 = (__int64)&off_3F92B2E;
  v31 = **(_QWORD **)(v4 - 24);
  v32 = sub_1648B60(64);
  v36 = v32;
  if ( v32 )
  {
    v89 = v32;
    v96 = v32;
    sub_15F1EA0(v32, v31, 53, 0, 0, 0);
    *(_DWORD *)(v96 + 56) = v29;
    sub_164B780(v96, &v104);
    v27 = *(unsigned int *)(v96 + 56);
    sub_1648880(v96, v27, 1);
    v36 = v96;
  }
  else
  {
    v89 = 0;
  }
  v97 = *(_QWORD *)(v4 - 24);
  if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
    v37 = *(_QWORD *)(v2 - 8);
  else
    v37 = v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF);
  v38 = *(_QWORD *)(v37 + 24LL * *(unsigned int *)(v2 + 56) + 8);
  v39 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
  if ( v39 == *(_DWORD *)(v36 + 56) )
  {
    v87 = v36;
    sub_15F55D0(v36, v27, v37, v33, v34, v35);
    v36 = v87;
    v39 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
  }
  v40 = (v39 + 1) & 0xFFFFFFF;
  v41 = v40 | *(_DWORD *)(v36 + 20) & 0xF0000000;
  *(_DWORD *)(v36 + 20) = v41;
  if ( (v41 & 0x40000000) != 0 )
    v42 = *(_QWORD *)(v36 - 8);
  else
    v42 = v89 - 24 * v40;
  v43 = (_QWORD *)(v42 + 24LL * (unsigned int)(v40 - 1));
  if ( *v43 )
  {
    v44 = v43[1];
    v45 = v43[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v45 = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 16) = v45 | *(_QWORD *)(v44 + 16) & 3LL;
  }
  *v43 = v97;
  if ( v97 )
  {
    v46 = *(_QWORD *)(v97 + 8);
    v43[1] = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = (unsigned __int64)(v43 + 1) | *(_QWORD *)(v46 + 16) & 3LL;
    v43[2] = (v97 + 8) | v43[2] & 3LL;
    *(_QWORD *)(v97 + 8) = v43;
  }
  v47 = *(_DWORD *)(v36 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v36 + 23) & 0x40) != 0 )
    v48 = *(_QWORD *)(v36 - 8);
  else
    v48 = v89 - 24 * v47;
  v84 = v36;
  *(_QWORD *)(v48 + 8LL * (unsigned int)(v47 - 1) + 24LL * *(unsigned int *)(v36 + 56) + 8) = v38;
  LOWORD(v106) = 257;
  v49 = sub_1648A60(64, 1u);
  v50 = v84;
  v12 = (__int64)v49;
  if ( v49 )
  {
    sub_15F90A0((__int64)v49, *(_QWORD *)(*(_QWORD *)v84 + 24LL), v84, (__int64)&v104, v90 & 1, v92, 0);
    v50 = v84;
  }
  v85 = v2;
  v51 = (char *)&v104 + 4;
  v104 = 0x400000001LL;
  v52 = 1;
  v105 = 0x700000006LL;
  v106 = 0xB00000008LL;
  v107 = 0xC00000011LL;
  v108 = 13;
  v93 = v50;
  while ( 1 )
  {
    v53 = *(_QWORD *)(v4 + 48);
    if ( v53 || *(__int16 *)(v4 + 18) < 0 )
      v53 = sub_1625790(v4, v52);
    sub_1625C10(v12, v52, v53);
    if ( &v109 == v51 )
      break;
    v52 = *(_DWORD *)v51;
    v51 += 4;
  }
  v55 = v85;
  v56 = v93;
  v86 = *(_DWORD *)(v85 + 20) & 0xFFFFFFF;
  if ( v86 != 1 )
  {
    v57 = v97;
    for ( j = 1; j != v86; ++j )
    {
      if ( (*(_BYTE *)(v55 + 23) & 0x40) != 0 )
        v59 = *(_QWORD *)(v55 - 8);
      else
        v59 = v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF);
      v98 = *(_QWORD *)(v59 + 24LL * j);
      sub_1AEC0C0(v12, v98, &v104, 9);
      v62 = *(_QWORD *)(v98 - 24);
      if ( v62 != v57 )
        v57 = 0;
      if ( (*(_BYTE *)(v55 + 23) & 0x40) != 0 )
        v63 = *(_QWORD *)(v55 - 8);
      else
        v63 = v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF);
      v64 = *(_QWORD *)(v63 + 8LL * j + 24LL * *(unsigned int *)(v55 + 56) + 8);
      v65 = *(_DWORD *)(v93 + 20) & 0xFFFFFFF;
      if ( v65 == *(_DWORD *)(v93 + 56) )
      {
        v82 = *(_QWORD *)(v63 + 8LL * j + 24LL * *(unsigned int *)(v55 + 56) + 8);
        v99 = *(_QWORD *)(v98 - 24);
        sub_15F55D0(v93, v63, v62, 3LL * *(unsigned int *)(v55 + 56), v60, v61);
        v64 = v82;
        v62 = v99;
        v65 = *(_DWORD *)(v93 + 20) & 0xFFFFFFF;
      }
      v66 = (v65 + 1) & 0xFFFFFFF;
      v67 = v66 | *(_DWORD *)(v93 + 20) & 0xF0000000;
      *(_DWORD *)(v93 + 20) = v67;
      if ( (v67 & 0x40000000) != 0 )
        v68 = *(_QWORD *)(v93 - 8);
      else
        v68 = v89 - 24 * v66;
      v69 = (__int64 *)(v68 + 24LL * (unsigned int)(v66 - 1));
      if ( *v69 )
      {
        v70 = v69[1];
        v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v71 = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
      }
      *v69 = v62;
      if ( v62 )
      {
        v72 = *(_QWORD *)(v62 + 8);
        v69[1] = v72;
        if ( v72 )
          *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
        v69[2] = v69[2] & 3 | (v62 + 8);
        *(_QWORD *)(v62 + 8) = v69;
      }
      v73 = *(_DWORD *)(v93 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v93 + 23) & 0x40) != 0 )
        v74 = *(_QWORD *)(v93 - 8);
      else
        v74 = v89 - 24 * v73;
      *(_QWORD *)(v74 + 8LL * (unsigned int)(v73 - 1) + 24LL * *(unsigned int *)(v93 + 56) + 8) = v64;
    }
    v97 = v57;
    v56 = v93;
  }
  if ( v97 )
  {
    if ( *(_QWORD *)(v12 - 24) )
    {
      v75 = *(_QWORD *)(v12 - 16);
      v76 = *(_QWORD *)(v12 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v76 = v75;
      if ( v75 )
        *(_QWORD *)(v75 + 16) = v76 | *(_QWORD *)(v75 + 16) & 3LL;
    }
    *(_QWORD *)(v12 - 24) = v97;
    v77 = *(_QWORD *)(v97 + 8);
    *(_QWORD *)(v12 - 16) = v77;
    if ( v77 )
      *(_QWORD *)(v77 + 16) = (v12 - 16) | *(_QWORD *)(v77 + 16) & 3LL;
    v94 = v56;
    *(_QWORD *)(v12 - 8) = (v97 + 8) | *(_QWORD *)(v12 - 8) & 3LL;
    *(_QWORD *)(v97 + 8) = v12 - 24;
    sub_15F2000(v89);
    sub_1648B90(v94);
  }
  else
  {
    v100 = v56;
    sub_157E9D0(*(_QWORD *)(v55 + 40) + 40LL, v56);
    v81 = *(_QWORD *)(v55 + 24);
    *(_QWORD *)(v100 + 32) = v55 + 24;
    v81 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v100 + 24) = v81 | *(_QWORD *)(v100 + 24) & 7LL;
    *(_QWORD *)(v81 + 8) = v100 + 24;
    *(_QWORD *)(v55 + 24) = *(_QWORD *)(v55 + 24) & 7LL | (v100 + 24);
    sub_170B990(*a1, v100);
  }
  if ( v90 )
  {
    v78 = 3LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF);
    v79 = (__int64 *)(v55 - v78 * 8);
    if ( (*(_BYTE *)(v55 + 23) & 0x40) != 0 )
    {
      v79 = *(__int64 **)(v55 - 8);
      v102 = &v79[v78];
    }
    for ( ; v102 != v79; *(_WORD *)(v80 + 18) &= ~1u )
    {
      v80 = *v79;
      v79 += 3;
    }
  }
  sub_1789760((__int64)a1, v12, v55);
  return v12;
}
