// Function: sub_1B889F0
// Address: 0x1b889f0
//
__int64 *__fastcall sub_1B889F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  _QWORD *v7; // rsi
  int v8; // r9d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 *v13; // r14
  unsigned __int64 v14; // r8
  __int64 *v15; // r13
  int v16; // edx
  __int64 *v17; // rax
  int v18; // r15d
  __int64 v19; // r14
  __int64 *v20; // rax
  _QWORD *v21; // rax
  unsigned int v22; // r8d
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // rax
  int v27; // r8d
  __int64 *v28; // r10
  __int64 *v29; // rcx
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rax
  _QWORD *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 **v38; // rax
  __int64 *v39; // rax
  __int64 **v40; // rdx
  __int64 *v41; // r13
  __int64 *v42; // rsi
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned int v45; // esi
  __int64 v46; // r10
  __int64 v47; // rax
  unsigned int v48; // edx
  unsigned int v49; // r8d
  __int64 *v50; // rdi
  __int64 v51; // rcx
  unsigned int v52; // r8d
  __int64 **v53; // rdi
  __int64 *v54; // rcx
  unsigned int v55; // eax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 *v61; // rax
  int v62; // ebx
  __int64 **v63; // r9
  int v64; // eax
  int v65; // r11d
  __int64 *v66; // r9
  int v67; // eax
  int v68; // eax
  int v69; // esi
  __int64 v70; // r8
  unsigned int v71; // edx
  __int64 *v72; // rcx
  int v73; // r10d
  __int64 **v74; // rdi
  int v75; // ecx
  int v76; // ecx
  __int64 v77; // r11
  unsigned int v78; // edx
  __int64 v79; // rsi
  int v80; // r8d
  __int64 *v81; // rdi
  int v82; // edx
  int v83; // edx
  __int64 v84; // rdi
  __int64 *v85; // rsi
  unsigned int v86; // r14d
  int v87; // r8d
  __int64 v88; // rcx
  int v89; // eax
  int v90; // ecx
  __int64 v91; // r8
  __int64 **v92; // rsi
  unsigned int v93; // r14d
  int v94; // edi
  __int64 *v95; // rdx
  _QWORD *v96; // [rsp+8h] [rbp-E8h]
  int v97; // [rsp+10h] [rbp-E0h]
  int v98; // [rsp+14h] [rbp-DCh]
  __int64 v99; // [rsp+18h] [rbp-D8h]
  __int64 v100; // [rsp+20h] [rbp-D0h]
  __int64 v101; // [rsp+28h] [rbp-C8h]
  __int64 v102; // [rsp+38h] [rbp-B8h]
  __int64 v104; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v105; // [rsp+48h] [rbp-A8h]
  _QWORD v106[2]; // [rsp+50h] [rbp-A0h] BYREF
  char v107; // [rsp+60h] [rbp-90h]
  char v108; // [rsp+61h] [rbp-8Fh]
  __int64 *v109; // [rsp+70h] [rbp-80h] BYREF
  __int64 v110; // [rsp+78h] [rbp-78h]
  _BYTE v111[112]; // [rsp+80h] [rbp-70h] BYREF

  v3 = sub_1B7F330(a1, a2);
  v4 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(v3 - 8);
  else
    v5 = v3 - 24 * v4;
  v99 = 3LL * (unsigned int)(v4 - 1);
  v6 = *(_QWORD *)(v5 + v99 * 8);
  if ( *(_BYTE *)(v6 + 16) != 13 )
    BUG();
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v101 = sub_15A0680(*(_QWORD *)v6, (__int64)v7 - 1, 0);
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
  {
    v9 = *(_QWORD *)(v3 - 8);
    v10 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    v11 = (__int64 *)(v9 + v10);
  }
  else
  {
    v11 = (__int64 *)v3;
    v10 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    v9 = v3 - v10;
  }
  v12 = v10 - 24;
  v13 = (__int64 *)(v9 + 24);
  v109 = (__int64 *)v111;
  v110 = 0x800000000LL;
  v14 = 0xAAAAAAAAAAAAAAABLL * (v12 >> 3);
  if ( (unsigned __int64)v12 > 0xC0 )
  {
    v105 = 0xAAAAAAAAAAAAAAABLL * (v12 >> 3);
    sub_16CD150((__int64)&v109, v111, v105, 8, v14, v8);
    v15 = v109;
    v16 = v110;
    LODWORD(v14) = v105;
    v17 = &v109[(unsigned int)v110];
  }
  else
  {
    v15 = (__int64 *)v111;
    v16 = 0;
    v17 = (__int64 *)v111;
  }
  if ( v13 != v11 )
  {
    do
    {
      if ( v17 )
        *v17 = *v13;
      v13 += 3;
      ++v17;
    }
    while ( v13 != v11 );
    v15 = v109;
    v16 = v110;
  }
  v18 = v14 + v16;
  v108 = 1;
  LODWORD(v110) = v14 + v16;
  v19 = (unsigned int)(v14 + v16);
  v106[0] = "GapLoadGEP";
  v107 = 3;
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    v20 = *(__int64 **)(v3 - 8);
  else
    v20 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
  v104 = *v20;
  v102 = sub_16348C0(v3);
  if ( !v102 )
  {
    v59 = *(_QWORD *)v104;
    if ( *(_BYTE *)(*(_QWORD *)v104 + 8LL) == 16 )
      v60 = *(_QWORD *)(**(_QWORD **)(v59 + 16) + 24LL);
    else
      v60 = *(_QWORD *)(v59 + 24);
    v102 = v60;
  }
  v21 = sub_1648A60(72, v18 + 1);
  v22 = v18 + 1;
  v23 = (__int64)v21;
  if ( v21 )
  {
    v96 = v21;
    v100 = (__int64)&v21[-3 * v22];
    v24 = *(_QWORD *)v104;
    if ( *(_BYTE *)(*(_QWORD *)v104 + 8LL) == 16 )
      v24 = **(_QWORD **)(v24 + 16);
    v97 = v22;
    v98 = *(_DWORD *)(v24 + 8) >> 8;
    v25 = (__int64 *)sub_15F9F50(v102, (__int64)v15, v19);
    v26 = (__int64 *)sub_1646BA0(v25, v98);
    v27 = v97;
    v28 = v26;
    if ( *(_BYTE *)(*(_QWORD *)v104 + 8LL) == 16 )
    {
      v61 = sub_16463B0(v26, *(_QWORD *)(*(_QWORD *)v104 + 32LL));
      v27 = v97;
      v28 = v61;
    }
    else
    {
      v29 = &v15[v19];
      if ( v29 != v15 )
      {
        v30 = v15;
        while ( 1 )
        {
          v31 = *(_QWORD *)*v30;
          if ( *(_BYTE *)(v31 + 8) == 16 )
            break;
          if ( v29 == ++v30 )
            goto LABEL_27;
        }
        v32 = sub_16463B0(v28, *(_QWORD *)(v31 + 32));
        v27 = v97;
        v28 = v32;
      }
    }
LABEL_27:
    sub_15F1EA0(v23, (__int64)v28, 32, v100, v27, 0);
    *(_QWORD *)(v23 + 56) = v102;
    *(_QWORD *)(v23 + 64) = sub_15F9F50(v102, (__int64)v15, v19);
    sub_15F9CE0(v23, v104, v15, v19, (__int64)v106);
  }
  else
  {
    v96 = 0;
  }
  sub_15FA2E0(v23, (*(_BYTE *)(v3 + 17) & 2) != 0);
  if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
    v33 = *(_QWORD **)(v23 - 8);
  else
    v33 = &v96[-3 * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF)];
  v34 = &v33[v99];
  if ( *v34 )
  {
    v35 = v34[1];
    v36 = v34[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v36 = v35;
    if ( v35 )
      *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
  }
  *v34 = v101;
  if ( v101 )
  {
    v37 = *(_QWORD *)(v101 + 8);
    v34[1] = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = (unsigned __int64)(v34 + 1) | *(_QWORD *)(v37 + 16) & 3LL;
    v34[2] = (v101 + 8) | v34[2] & 3;
    *(_QWORD *)(v101 + 8) = v34;
  }
  sub_15F2120(v23, a2);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v38 = *(__int64 ***)(a2 - 8);
    if ( **v38 != *(_QWORD *)v23 )
    {
      v108 = 1;
      v106[0] = "GapLoadCast";
      v107 = 3;
      v39 = *v38;
      goto LABEL_46;
    }
  }
  else
  {
    v40 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v39 = *v40;
    if ( *(_QWORD *)v23 != **v40 )
    {
      v108 = 1;
      v106[0] = "GapLoadCast";
      v107 = 3;
LABEL_46:
      v23 = sub_15FDF90(v23, *v39, (__int64)v106, 0);
      sub_15F2120(v23, a2);
      v41 = (__int64 *)sub_15F4880(a2);
      v42 = v41 - 3;
      if ( !*(v41 - 3) )
        goto LABEL_49;
      goto LABEL_47;
    }
  }
  v57 = sub_15F4880(a2);
  v41 = (__int64 *)v57;
  v42 = (__int64 *)(v57 - 24);
  if ( *(_QWORD *)(v57 - 24) )
  {
LABEL_47:
    v43 = *(v41 - 2);
    v44 = *(v41 - 1) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v44 = v43;
    if ( v43 )
      *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
LABEL_49:
    *(v41 - 3) = v23;
    if ( !v23 )
      goto LABEL_50;
    goto LABEL_58;
  }
  *(_QWORD *)(v57 - 24) = v23;
LABEL_58:
  v58 = *(_QWORD *)(v23 + 8);
  *(v41 - 2) = v58;
  if ( v58 )
    *(_QWORD *)(v58 + 16) = (unsigned __int64)(v41 - 2) | *(_QWORD *)(v58 + 16) & 3LL;
  *(v41 - 1) = (v23 + 8) | *(v41 - 1) & 3;
  *(_QWORD *)(v23 + 8) = v42;
LABEL_50:
  v45 = *(_DWORD *)(a1 + 264);
  v46 = a1 + 240;
  if ( !v45 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_95;
  }
  v47 = *(_QWORD *)(a1 + 248);
  v48 = v45 - 1;
  v49 = (v45 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v50 = (__int64 *)(v47 + 8LL * v49);
  v51 = *v50;
  if ( v23 == *v50 )
    goto LABEL_52;
  v65 = 1;
  v66 = 0;
  while ( v51 != -8 )
  {
    if ( v66 || v51 != -16 )
      v50 = v66;
    v49 = v48 & (v65 + v49);
    v51 = *(_QWORD *)(v47 + 8LL * v49);
    if ( v23 == v51 )
      goto LABEL_52;
    ++v65;
    v66 = v50;
    v50 = (__int64 *)(v47 + 8LL * v49);
  }
  if ( !v66 )
    v66 = v50;
  ++*(_QWORD *)(a1 + 240);
  v67 = *(_DWORD *)(a1 + 256) + 1;
  if ( 4 * v67 >= 3 * v45 )
  {
LABEL_95:
    sub_1467110(v46, 2 * v45);
    v75 = *(_DWORD *)(a1 + 264);
    if ( v75 )
    {
      v76 = v75 - 1;
      v77 = *(_QWORD *)(a1 + 248);
      v46 = a1 + 240;
      v78 = v76 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v66 = (__int64 *)(v77 + 8LL * v78);
      v79 = *v66;
      v67 = *(_DWORD *)(a1 + 256) + 1;
      if ( v23 != *v66 )
      {
        v80 = 1;
        v81 = 0;
        while ( v79 != -8 )
        {
          if ( !v81 && v79 == -16 )
            v81 = v66;
          v78 = v76 & (v80 + v78);
          v66 = (__int64 *)(v77 + 8LL * v78);
          v79 = *v66;
          if ( v23 == *v66 )
            goto LABEL_82;
          ++v80;
        }
        if ( v81 )
          v66 = v81;
      }
      goto LABEL_82;
    }
    goto LABEL_146;
  }
  if ( v45 - *(_DWORD *)(a1 + 260) - v67 <= v45 >> 3 )
  {
    sub_1467110(v46, v45);
    v82 = *(_DWORD *)(a1 + 264);
    if ( v82 )
    {
      v83 = v82 - 1;
      v84 = *(_QWORD *)(a1 + 248);
      v85 = 0;
      v46 = a1 + 240;
      v86 = v83 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v87 = 1;
      v66 = (__int64 *)(v84 + 8LL * v86);
      v88 = *v66;
      v67 = *(_DWORD *)(a1 + 256) + 1;
      if ( v23 != *v66 )
      {
        while ( v88 != -8 )
        {
          if ( v88 == -16 && !v85 )
            v85 = v66;
          v86 = v83 & (v87 + v86);
          v66 = (__int64 *)(v84 + 8LL * v86);
          v88 = *v66;
          if ( v23 == *v66 )
            goto LABEL_82;
          ++v87;
        }
        if ( v85 )
          v66 = v85;
      }
      goto LABEL_82;
    }
LABEL_146:
    ++*(_DWORD *)(a1 + 256);
    BUG();
  }
LABEL_82:
  *(_DWORD *)(a1 + 256) = v67;
  if ( *v66 != -8 )
    --*(_DWORD *)(a1 + 260);
  *v66 = v23;
  v45 = *(_DWORD *)(a1 + 264);
  v47 = *(_QWORD *)(a1 + 248);
  if ( !v45 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_86;
  }
  v48 = v45 - 1;
LABEL_52:
  v52 = v48 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
  v53 = (__int64 **)(v47 + 8LL * v52);
  v54 = *v53;
  if ( *v53 == v41 )
    goto LABEL_53;
  v62 = 1;
  v63 = 0;
  while ( v54 != (__int64 *)-8LL )
  {
    if ( v54 != (__int64 *)-16LL || v63 )
      v53 = v63;
    v52 = v48 & (v62 + v52);
    v54 = *(__int64 **)(v47 + 8LL * v52);
    if ( v54 == v41 )
      goto LABEL_53;
    ++v62;
    v63 = v53;
    v53 = (__int64 **)(v47 + 8LL * v52);
  }
  if ( !v63 )
    v63 = v53;
  ++*(_QWORD *)(a1 + 240);
  v64 = *(_DWORD *)(a1 + 256) + 1;
  if ( 4 * v64 >= 3 * v45 )
  {
LABEL_86:
    sub_1467110(v46, 2 * v45);
    v68 = *(_DWORD *)(a1 + 264);
    if ( v68 )
    {
      v69 = v68 - 1;
      v70 = *(_QWORD *)(a1 + 248);
      v71 = (v68 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v63 = (__int64 **)(v70 + 8LL * v71);
      v72 = *v63;
      v64 = *(_DWORD *)(a1 + 256) + 1;
      if ( *v63 != v41 )
      {
        v73 = 1;
        v74 = 0;
        while ( v72 != (__int64 *)-8LL )
        {
          if ( v72 == (__int64 *)-16LL && !v74 )
            v74 = v63;
          v71 = v69 & (v73 + v71);
          v63 = (__int64 **)(v70 + 8LL * v71);
          v72 = *v63;
          if ( *v63 == v41 )
            goto LABEL_73;
          ++v73;
        }
        if ( v74 )
          v63 = v74;
      }
      goto LABEL_73;
    }
    goto LABEL_147;
  }
  if ( v45 - (v64 + *(_DWORD *)(a1 + 260)) <= v45 >> 3 )
  {
    sub_1467110(v46, v45);
    v89 = *(_DWORD *)(a1 + 264);
    if ( v89 )
    {
      v90 = v89 - 1;
      v91 = *(_QWORD *)(a1 + 248);
      v92 = 0;
      v93 = (v89 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v63 = (__int64 **)(v91 + 8LL * v93);
      v94 = 1;
      v95 = *v63;
      v64 = *(_DWORD *)(a1 + 256) + 1;
      if ( *v63 != v41 )
      {
        while ( v95 != (__int64 *)-8LL )
        {
          if ( !v92 && v95 == (__int64 *)-16LL )
            v92 = v63;
          v93 = v90 & (v94 + v93);
          v63 = (__int64 **)(v91 + 8LL * v93);
          v95 = *v63;
          if ( *v63 == v41 )
            goto LABEL_73;
          ++v94;
        }
        if ( v92 )
          v63 = v92;
      }
      goto LABEL_73;
    }
LABEL_147:
    ++*(_DWORD *)(a1 + 256);
    BUG();
  }
LABEL_73:
  *(_DWORD *)(a1 + 256) = v64;
  if ( *v63 != (__int64 *)-8LL )
    --*(_DWORD *)(a1 + 260);
  *v63 = v41;
LABEL_53:
  v55 = sub_16431D0(*v41);
  sub_15F8F50((__int64)v41, v55 >> 3);
  sub_15F2120((__int64)v41, a2);
  if ( v109 != (__int64 *)v111 )
    _libc_free((unsigned __int64)v109);
  return v41;
}
