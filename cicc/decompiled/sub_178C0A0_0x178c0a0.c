// Function: sub_178C0A0
// Address: 0x178c0a0
//
__int64 __fastcall sub_178C0A0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // r13
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int8 v11; // al
  __int64 **v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned int v17; // ebx
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  char v23; // al
  _QWORD **v24; // r15
  const char *v25; // rax
  int v26; // r15d
  int v27; // r15d
  __int64 v28; // rdx
  __int64 **v29; // rax
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rsi
  __int64 v35; // r10
  __int64 v36; // r11
  __int64 *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // r15
  int v41; // eax
  __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rdi
  unsigned __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r9
  __int64 v52; // rbx
  unsigned int v53; // r15d
  __int64 v54; // r13
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 *v57; // rdx
  __int64 v58; // r14
  __int64 v59; // r8
  int v60; // eax
  __int64 v61; // rax
  int v62; // edx
  __int64 v63; // rdx
  _QWORD *v64; // rax
  __int64 v65; // rcx
  unsigned __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v71; // rax
  __int64 v72; // r15
  char v73; // al
  int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned __int8 *v77; // rax
  __int64 *v78; // rcx
  unsigned int v79; // ebx
  int v80; // r14d
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rcx
  __int64 *v84; // [rsp+8h] [rbp-98h]
  __int64 v85; // [rsp+10h] [rbp-90h]
  __int64 v86; // [rsp+20h] [rbp-80h]
  __int64 v87; // [rsp+20h] [rbp-80h]
  __int64 v88; // [rsp+28h] [rbp-78h]
  unsigned int v89; // [rsp+28h] [rbp-78h]
  __int64 v90; // [rsp+28h] [rbp-78h]
  __int64 v91; // [rsp+30h] [rbp-70h]
  int v92; // [rsp+38h] [rbp-68h]
  __int64 v93; // [rsp+38h] [rbp-68h]
  __int64 v94; // [rsp+38h] [rbp-68h]
  __int64 v95; // [rsp+38h] [rbp-68h]
  __int64 v96; // [rsp+38h] [rbp-68h]
  __int64 *v97; // [rsp+38h] [rbp-68h]
  __int64 v98; // [rsp+38h] [rbp-68h]
  unsigned __int8 *v99; // [rsp+38h] [rbp-68h]
  __int64 v100; // [rsp+38h] [rbp-68h]
  _QWORD v101[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v102[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v103; // [rsp+60h] [rbp-40h]

  v2 = a1;
  v4 = sub_157EBA0(*(_QWORD *)(a2 + 40));
  if ( v4 )
  {
    v9 = (unsigned int)*(unsigned __int8 *)(v4 + 16) - 34;
    if ( (unsigned int)v9 <= 0x36 )
    {
      v5 = 0x40018000000001LL;
      if ( _bittest64(&v5, v9) )
        return 0;
    }
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
  {
    v5 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v10 = *(_QWORD *)(a2 - v5);
    v11 = *(_BYTE *)(v10 + 16);
    if ( v11 != 56 )
      goto LABEL_6;
    return sub_1789990(a1, a2, v5, v6, v7, v8);
  }
  v10 = **(_QWORD **)(a2 - 8);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 == 56 )
    return sub_1789990(a1, a2, v5, v6, v7, v8);
LABEL_6:
  if ( v11 == 54 )
    return sub_178B660(a1, a2);
  if ( (unsigned int)v11 - 60 > 0xC )
  {
    if ( (unsigned __int8)(v11 - 75) <= 1u || (unsigned int)v11 - 35 <= 0x11 )
    {
      if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
        v71 = *(_QWORD *)(v10 - 8);
      else
        v71 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      v15 = 0;
      v91 = *(_QWORD *)(v71 + 24);
      if ( *(_BYTE *)(v91 + 16) > 0x10u )
        return sub_178AA50(a1, a2);
      goto LABEL_11;
    }
    return 0;
  }
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v12 = *(__int64 ***)(v10 - 8);
  else
    v12 = (__int64 **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v13 = *(_QWORD *)a2;
  v14 = *v12;
  v91 = 0;
  v15 = *v14;
  if ( *(_BYTE *)(v13 + 8) == 11 && *(_BYTE *)(v15 + 8) == 11 )
  {
    v95 = *v14;
    v73 = sub_1705440((__int64)a1, v13, *v14);
    v15 = v95;
    if ( !v73 )
      return 0;
  }
LABEL_11:
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
  {
    v92 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v16 = v10;
    v17 = 1;
    v18 = v15;
    v19 = v16;
    while ( 1 )
    {
      v20 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v21 = *(_QWORD *)(v20 + 24LL * v17);
      if ( *(_BYTE *)(v21 + 16) <= 0x17u )
        return 0;
      v22 = *(_QWORD *)(v21 + 8);
      if ( !v22 || *(_QWORD *)(v22 + 8) || !sub_15F4220(v21, v19, 0) )
        return 0;
      v23 = *(_BYTE *)(v21 + 23) & 0x40;
      if ( v18 )
      {
        if ( v23 )
          v24 = *(_QWORD ***)(v21 - 8);
        else
          v24 = (_QWORD **)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
        if ( v18 != **v24 )
          return 0;
      }
      else
      {
        if ( v23 )
          v72 = *(_QWORD *)(v21 - 8);
        else
          v72 = v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
        if ( v91 != *(_QWORD *)(v72 + 24) )
          return 0;
      }
      if ( v92 == ++v17 )
      {
        v2 = a1;
        v10 = v19;
        break;
      }
    }
  }
  v25 = sub_1649960(a2);
  v26 = *(_DWORD *)(a2 + 20);
  v101[0] = v25;
  v102[0] = (__int64)v101;
  v27 = v26 & 0xFFFFFFF;
  v101[1] = v28;
  v103 = 773;
  v102[1] = (__int64)&off_3F92B2E;
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v29 = *(__int64 ***)(v10 - 8);
  else
    v29 = (__int64 **)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v93 = **v29;
  v30 = sub_1648B60(64);
  v34 = v93;
  v35 = v30;
  if ( v30 )
  {
    v88 = v30;
    v94 = v30;
    sub_15F1EA0(v30, v34, 53, 0, 0, 0);
    *(_DWORD *)(v94 + 56) = v27;
    sub_164B780(v94, v102);
    sub_1648880(v94, *(_DWORD *)(v94 + 56), 1);
    v36 = v88;
    v35 = v94;
  }
  else
  {
    v36 = 0;
  }
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v37 = *(__int64 **)(v10 - 8);
  else
    v37 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
  v38 = *v37;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v39 = *(_QWORD *)(a2 - 8);
  else
    v39 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v40 = *(_QWORD *)(v39 + 24LL * *(unsigned int *)(a2 + 56) + 8);
  v41 = *(_DWORD *)(v35 + 20) & 0xFFFFFFF;
  if ( v41 == *(_DWORD *)(v35 + 56) )
  {
    v87 = v36;
    v100 = v35;
    sub_15F55D0(v35, v38, v39, v31, v32, v33);
    v35 = v100;
    v36 = v87;
    v41 = *(_DWORD *)(v100 + 20) & 0xFFFFFFF;
  }
  v42 = (v41 + 1) & 0xFFFFFFF;
  v43 = v42 | *(_DWORD *)(v35 + 20) & 0xF0000000;
  *(_DWORD *)(v35 + 20) = v43;
  if ( (v43 & 0x40000000) != 0 )
    v44 = *(_QWORD *)(v35 - 8);
  else
    v44 = v36 - 24 * v42;
  v45 = (__int64 *)(v44 + 24LL * (unsigned int)(v42 - 1));
  if ( *v45 )
  {
    v46 = v45[1];
    v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v47 = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
  }
  *v45 = v38;
  if ( v38 )
  {
    v48 = *(_QWORD *)(v38 + 8);
    v45[1] = v48;
    if ( v48 )
      *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
    v45[2] = (v38 + 8) | v45[2] & 3;
    *(_QWORD *)(v38 + 8) = v45;
  }
  v49 = *(_DWORD *)(v35 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v35 + 23) & 0x40) != 0 )
    v50 = *(_QWORD *)(v35 - 8);
  else
    v50 = v36 - 24 * v49;
  *(_QWORD *)(v50 + 8LL * (unsigned int)(v49 - 1) + 24LL * *(unsigned int *)(v35 + 56) + 8) = v40;
  v51 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v51 != 1 )
  {
    v85 = v10;
    v52 = v35;
    v84 = v2;
    v53 = 1;
    v54 = v38;
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v55 = *(_QWORD *)(a2 - 8);
      else
        v55 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v56 = *(_QWORD *)(v55 + 24LL * v53);
      if ( (*(_BYTE *)(v56 + 23) & 0x40) != 0 )
        v57 = *(__int64 **)(v56 - 8);
      else
        v57 = (__int64 *)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
      v58 = *v57;
      if ( *v57 != v54 )
        v54 = 0;
      v59 = *(_QWORD *)(v55 + 8LL * v53 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v60 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
      if ( v60 == *(_DWORD *)(v52 + 56) )
      {
        v86 = v36;
        v89 = v51;
        v96 = *(_QWORD *)(v55 + 8LL * v53 + 24LL * *(unsigned int *)(a2 + 56) + 8);
        sub_15F55D0(v52, 0, 3LL * *(unsigned int *)(a2 + 56), v55, v59, v51);
        v36 = v86;
        v51 = v89;
        v59 = v96;
        v60 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
      }
      v61 = (v60 + 1) & 0xFFFFFFF;
      v62 = v61 | *(_DWORD *)(v52 + 20) & 0xF0000000;
      *(_DWORD *)(v52 + 20) = v62;
      if ( (v62 & 0x40000000) != 0 )
        v63 = *(_QWORD *)(v52 - 8);
      else
        v63 = v36 - 24 * v61;
      v64 = (_QWORD *)(v63 + 24LL * (unsigned int)(v61 - 1));
      if ( *v64 )
      {
        v65 = v64[1];
        v66 = v64[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v66 = v65;
        if ( v65 )
          *(_QWORD *)(v65 + 16) = *(_QWORD *)(v65 + 16) & 3LL | v66;
      }
      *v64 = v58;
      if ( v58 )
      {
        v67 = *(_QWORD *)(v58 + 8);
        v64[1] = v67;
        if ( v67 )
          *(_QWORD *)(v67 + 16) = (unsigned __int64)(v64 + 1) | *(_QWORD *)(v67 + 16) & 3LL;
        v64[2] = v64[2] & 3LL | (v58 + 8);
        *(_QWORD *)(v58 + 8) = v64;
      }
      v68 = *(_DWORD *)(v52 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v52 + 23) & 0x40) != 0 )
        v69 = *(_QWORD *)(v52 - 8);
      else
        v69 = v36 - 24 * v68;
      ++v53;
      *(_QWORD *)(v69 + 8LL * (unsigned int)(v68 - 1) + 24LL * *(unsigned int *)(v52 + 56) + 8) = v59;
    }
    while ( v53 != (_DWORD)v51 );
    v38 = v54;
    v35 = v52;
    v2 = v84;
    v10 = v85;
  }
  v97 = (__int64 *)v38;
  if ( v38 )
  {
    v90 = v35;
    sub_15F2000(v36);
    sub_1648B90(v90);
  }
  else
  {
    v97 = (__int64 *)v35;
    sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, v35);
    v83 = *(_QWORD *)(a2 + 24);
    v97[4] = a2 + 24;
    v83 &= 0xFFFFFFFFFFFFFFF8LL;
    v97[3] = v83 | v97[3] & 7;
    *(_QWORD *)(v83 + 8) = v97 + 3;
    *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (unsigned __int64)(v97 + 3);
    sub_170B990(*v2, (__int64)v97);
  }
  v74 = *(unsigned __int8 *)(v10 + 16);
  if ( (unsigned int)(v74 - 60) > 0xC )
  {
    v103 = 257;
    if ( (unsigned int)(v74 - 35) > 0x11 )
    {
      v76 = sub_15FEEB0(
              (unsigned int)*(unsigned __int8 *)(v10 + 16) - 24,
              *(_WORD *)(v10 + 18) & 0x7FFF,
              (__int64)v97,
              v91,
              (__int64)v102,
              0);
    }
    else
    {
      v77 = (unsigned __int8 *)sub_15FB440(
                                 (unsigned int)*(unsigned __int8 *)(v10 + 16) - 24,
                                 v97,
                                 v91,
                                 (__int64)v102,
                                 0);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v78 = *(__int64 **)(a2 - 8);
      else
        v78 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v99 = v77;
      v79 = 1;
      sub_15F2530(v77, *v78, 1);
      v76 = (__int64)v99;
      v80 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( v80 != 1 )
      {
        do
        {
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v81 = *(_QWORD *)(a2 - 8);
          else
            v81 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v82 = v79++;
          sub_15F2780(v99, *(_QWORD *)(v81 + 24 * v82));
        }
        while ( v79 != v80 );
        v76 = (__int64)v99;
      }
    }
  }
  else
  {
    v75 = *(_QWORD *)a2;
    v103 = 257;
    v76 = sub_15FDBD0((unsigned int)*(unsigned __int8 *)(v10 + 16) - 24, (__int64)v97, v75, (__int64)v102, 0);
  }
  v98 = v76;
  sub_1789760((__int64)v2, v76, a2);
  return v98;
}
