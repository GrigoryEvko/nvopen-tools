// Function: sub_13510F0
// Address: 0x13510f0
//
__int64 __fastcall sub_13510F0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  int v6; // r13d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // r12d
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // r8
  int v19; // r10d
  __int64 *v20; // r9
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // r15
  unsigned int i; // edx
  __int64 *v25; // r11
  __int64 v26; // rax
  unsigned int v27; // edx
  char v28; // al
  __int64 v29; // rdx
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // r12
  unsigned int v34; // r15d
  unsigned __int64 v35; // rbx
  unsigned int v36; // r13d
  __int64 v37; // rax
  char v38; // al
  unsigned __int64 v39; // rdi
  char v40; // dl
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  int v50; // edx
  unsigned int v51; // eax
  unsigned __int64 v52; // rdi
  __int64 v53; // rax
  int v54; // edx
  int v55; // ecx
  __int64 v56; // rdx
  int v57; // ecx
  int v58; // edi
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rsi
  int v61; // eax
  __int64 *v62; // rsi
  unsigned int v63; // eax
  __int64 v64; // r8
  unsigned int v65; // eax
  int v66; // edx
  int v67; // edx
  __int64 v68; // rdi
  int v69; // ecx
  unsigned int j; // r15d
  __int64 *v71; // rax
  __int64 v72; // rsi
  unsigned int v73; // r15d
  unsigned __int64 v74; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v75; // [rsp+0h] [rbp-D0h]
  __int64 v76; // [rsp+8h] [rbp-C8h]
  int v77; // [rsp+8h] [rbp-C8h]
  __int64 v78; // [rsp+8h] [rbp-C8h]
  int v79; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v80; // [rsp+8h] [rbp-C8h]
  int v81; // [rsp+10h] [rbp-C0h]
  __int64 *v82; // [rsp+18h] [rbp-B8h]
  unsigned __int8 v83; // [rsp+26h] [rbp-AAh]
  char v84; // [rsp+27h] [rbp-A9h]
  __int64 v86; // [rsp+38h] [rbp-98h] BYREF
  _QWORD v87[6]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v88[12]; // [rsp+70h] [rbp-60h] BYREF

  if ( !a4 )
    return 7;
  v6 = a4;
  v9 = sub_15F2050(a2);
  v10 = sub_1632FA0(v9);
  v11 = sub_14AD280(*a3, v10, 6);
  v84 = sub_134E860(v11);
  if ( !v84 )
    return 7;
  if ( *(_BYTE *)(v11 + 16) <= 0x10u )
    return 7;
  v14 = *(_BYTE *)(a2 + 16);
  if ( v14 <= 0x17u )
    return 7;
  v15 = a2 | 4;
  if ( v14 != 78 )
  {
    if ( v14 != 29 )
      return 7;
    v15 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v86 = v15;
  if ( (a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 || v11 == (a2 & 0xFFFFFFFFFFFFFFF8LL) )
    return 7;
  if ( !*(_BYTE *)(a1 + 1) )
  {
    v28 = sub_139D140(v11, 1, 1, a2, v6, 1, a5);
    goto LABEL_22;
  }
  v16 = *(_DWORD *)(a1 + 32);
  v17 = a1 + 8;
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_81;
  }
  v18 = *(_QWORD *)(a1 + 16);
  v19 = 1;
  v20 = 0;
  v21 = (((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
         | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
        | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32));
  v22 = ((v21 - 1 - (v21 << 13)) >> 8) ^ (v21 - 1 - (v21 << 13));
  v23 = (((((9 * v22) >> 15) ^ (9 * v22)) - 1 - ((((9 * v22) >> 15) ^ (9 * v22)) << 27)) >> 31)
      ^ ((((9 * v22) >> 15) ^ (9 * v22)) - 1 - ((((9 * v22) >> 15) ^ (9 * v22)) << 27));
  for ( i = v23 & (v16 - 1); ; i = (v16 - 1) & v27 )
  {
    v25 = (__int64 *)(v18 + 24LL * i);
    v26 = *v25;
    if ( v11 == *v25 && a2 == v25[1] )
    {
      v28 = *((_BYTE *)v25 + 16) ^ 1;
      goto LABEL_22;
    }
    if ( v26 == -8 )
      break;
    if ( v26 == -16 && v25[1] == -16 && !v20 )
      v20 = (__int64 *)(v18 + 24LL * i);
LABEL_20:
    v27 = v19 + i;
    ++v19;
  }
  if ( v25[1] != -8 )
    goto LABEL_20;
  if ( v20 )
    v25 = v20;
  ++*(_QWORD *)(a1 + 8);
  v54 = *(_DWORD *)(a1 + 24) + 1;
  if ( 4 * v54 < 3 * v16 )
  {
    if ( v16 - *(_DWORD *)(a1 + 28) - v54 > v16 >> 3 )
      goto LABEL_72;
    sub_1350E40(v17, v16);
    v66 = *(_DWORD *)(a1 + 32);
    if ( v66 )
    {
      v67 = v66 - 1;
      v68 = *(_QWORD *)(a1 + 16);
      v69 = 1;
      v25 = 0;
      for ( j = v67 & v23; ; j = v67 & v73 )
      {
        v71 = (__int64 *)(v68 + 24LL * j);
        v72 = *v71;
        if ( v11 == *v71 && a2 == v71[1] )
        {
          v25 = (__int64 *)(v68 + 24LL * j);
          v54 = *(_DWORD *)(a1 + 24) + 1;
          goto LABEL_72;
        }
        if ( v72 == -8 )
        {
          if ( v71[1] == -8 )
          {
            if ( !v25 )
              v25 = (__int64 *)(v68 + 24LL * j);
            v54 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_72;
          }
        }
        else if ( v72 == -16 && v71[1] == -16 && !v25 )
        {
          v25 = (__int64 *)(v68 + 24LL * j);
        }
        v73 = v69 + j;
        ++v69;
      }
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_81:
  sub_1350E40(v17, 2 * v16);
  v55 = *(_DWORD *)(a1 + 32);
  if ( !v55 )
    goto LABEL_113;
  v56 = *(_QWORD *)(a1 + 16);
  v57 = v55 - 1;
  v58 = 1;
  v59 = (((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
         | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
        | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32));
  v60 = ((9 * (((v59 - 1 - (v59 << 13)) >> 8) ^ (v59 - 1 - (v59 << 13)))) >> 15)
      ^ (9 * (((v59 - 1 - (v59 << 13)) >> 8) ^ (v59 - 1 - (v59 << 13))));
  v61 = ((v60 - 1 - (v60 << 27)) >> 31) ^ (v60 - 1 - ((_DWORD)v60 << 27));
  v62 = 0;
  v63 = v57 & v61;
  while ( 2 )
  {
    v25 = (__int64 *)(v56 + 24LL * v63);
    v64 = *v25;
    if ( v11 == *v25 && a2 == v25[1] )
    {
      v54 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_72;
    }
    if ( v64 != -8 )
    {
      if ( v64 == -16 && v25[1] == -16 && !v62 )
        v62 = (__int64 *)(v56 + 24LL * v63);
      goto LABEL_89;
    }
    if ( v25[1] != -8 )
    {
LABEL_89:
      v65 = v58 + v63;
      ++v58;
      v63 = v57 & v65;
      continue;
    }
    break;
  }
  if ( v62 )
    v25 = v62;
  v54 = *(_DWORD *)(a1 + 24) + 1;
LABEL_72:
  *(_DWORD *)(a1 + 24) = v54;
  if ( *v25 != -8 || v25[1] != -8 )
    --*(_DWORD *)(a1 + 28);
  *v25 = v11;
  v25[1] = a2;
  *((_BYTE *)v25 + 16) = 0;
  v82 = v25;
  v28 = sub_139D140(v11, 1, 1, a2, v6, 1, a5);
  *((_BYTE *)v82 + 16) = v28 ^ 1;
LABEL_22:
  if ( v28 )
    return 7;
  v29 = 24LL * (*(_DWORD *)((v86 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v30 = (v86 & 0xFFFFFFFFFFFFFFF8LL) - v29;
  if ( (*(_BYTE *)((v86 & 0xFFFFFFFFFFFFFFF8LL) + 23) & 0x40) != 0 )
    v30 = *(_QWORD *)((v86 & 0xFFFFFFFFFFFFFFF8LL) - 8);
  v31 = (-(__int64)(((v86 >> 2) & 1) == 0) & 0xFFFFFFFFFFFFFFD0LL) + v29 - 24;
  v32 = v30 + v31;
  if ( v30 == v30 + v31 )
    return 0;
  v83 = 4;
  v33 = v30;
  v34 = 0;
  v35 = v32;
  while ( 2 )
  {
    v36 = v34++;
    if ( *(_BYTE *)(**(_QWORD **)v33 + 8LL) != 15 )
      goto LABEL_27;
    if ( (unsigned __int8)sub_134FA60(&v86, v34, 22) )
    {
LABEL_30:
      v88[1] = -1;
      memset(&v88[2], 0, 24);
      v88[0] = v11;
      v37 = *(_QWORD *)v33;
      v87[1] = -1;
      v87[0] = v37;
      memset(&v87[2], 0, 24);
      v38 = sub_134CB50(a1, (__int64)v87, (__int64)v88);
      if ( v38 == 3 || (v84 = 0, v38) )
      {
        if ( !(unsigned __int8)sub_134FA60(&v86, v34, 36) )
        {
          v83 = 5;
          if ( !(unsigned __int8)sub_134FA60(&v86, v34, 37) && !(unsigned __int8)sub_134FA60(&v86, v34, 36) )
            return 7;
        }
      }
      goto LABEL_27;
    }
    v39 = v86 & 0xFFFFFFFFFFFFFFF8LL;
    v40 = *(_BYTE *)((v86 & 0xFFFFFFFFFFFFFFF8LL) + 23);
    v81 = *(_DWORD *)((v86 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
    if ( (v86 & 4) == 0 )
    {
      if ( v40 < 0 )
      {
        v75 = v86 & 0xFFFFFFFFFFFFFFF8LL;
        v46 = sub_1648A40(v39);
        v78 = v47 + v46;
        if ( *(char *)(v75 + 23) >= 0 )
        {
          if ( (unsigned int)(v78 >> 4) )
LABEL_117:
            BUG();
        }
        else if ( (unsigned int)((v78 - sub_1648A40(v75)) >> 4) )
        {
          if ( *(char *)(v75 + 23) >= 0 )
            goto LABEL_117;
          v79 = *(_DWORD *)(sub_1648A40(v75) + 8);
          if ( *(char *)(v75 + 23) >= 0 )
            BUG();
          v48 = sub_1648A40(v75);
          v50 = *(_DWORD *)(v48 + v49 - 4) - v79;
          goto LABEL_60;
        }
      }
      v50 = 0;
LABEL_60:
      v51 = v81 - 3 - v50;
      goto LABEL_52;
    }
    if ( v40 >= 0 )
      goto LABEL_50;
    v74 = v86 & 0xFFFFFFFFFFFFFFF8LL;
    v41 = sub_1648A40(v39);
    v76 = v42 + v41;
    if ( *(char *)(v74 + 23) >= 0 )
    {
      if ( (unsigned int)(v76 >> 4) )
LABEL_115:
        BUG();
      goto LABEL_50;
    }
    if ( !(unsigned int)((v76 - sub_1648A40(v74)) >> 4) )
    {
LABEL_50:
      v45 = 0;
      goto LABEL_51;
    }
    if ( *(char *)(v74 + 23) >= 0 )
      goto LABEL_115;
    v77 = *(_DWORD *)(sub_1648A40(v74) + 8);
    if ( *(char *)(v74 + 23) >= 0 )
      BUG();
    v43 = sub_1648A40(v74);
    v45 = *(_DWORD *)(v43 + v44 - 4) - v77;
LABEL_51:
    v51 = v81 - 1 - v45;
LABEL_52:
    if ( v36 >= v51 )
      goto LABEL_30;
    v80 = v86 & 0xFFFFFFFFFFFFFFF8LL;
    v52 = (v86 & 0xFFFFFFFFFFFFFFF8LL) + 56;
    if ( (v86 & 4) != 0 )
    {
      if ( (unsigned __int8)sub_1560290(v52, v36, 6) )
        goto LABEL_30;
      v53 = *(_QWORD *)(v80 - 24);
      if ( !*(_BYTE *)(v53 + 16) )
        goto LABEL_56;
    }
    else
    {
      if ( (unsigned __int8)sub_1560290(v52, v36, 6) )
        goto LABEL_30;
      v53 = *(_QWORD *)(v80 - 72);
      if ( !*(_BYTE *)(v53 + 16) )
      {
LABEL_56:
        v88[0] = *(_QWORD *)(v53 + 112);
        if ( (unsigned __int8)sub_1560290(v88, v36, 6) )
          goto LABEL_30;
      }
    }
LABEL_27:
    v33 += 24LL;
    if ( v33 != v35 )
      continue;
    break;
  }
  v12 = v83;
  if ( v84 )
    return v83 & 3;
  return v12;
}
