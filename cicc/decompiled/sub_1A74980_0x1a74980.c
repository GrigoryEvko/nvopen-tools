// Function: sub_1A74980
// Address: 0x1a74980
//
void __fastcall sub_1A74980(__int64 a1, char a2, __int64 a3)
{
  __int64 v5; // rcx
  __int64 *v6; // r13
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // rsi
  __int64 *v9; // rsi
  __int64 v10; // rdx
  unsigned __int64 *v11; // rdi
  unsigned __int64 *v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  _QWORD *v16; // rax
  int v17; // r8d
  _QWORD *v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  int v24; // r8d
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  __int64 v27; // r10
  __int64 *v28; // rax
  __int64 v29; // r14
  unsigned int v30; // ecx
  __int64 *v31; // rdx
  __int64 v32; // r10
  __int64 v33; // r8
  __int64 v34; // rax
  _BYTE *v35; // rax
  __int64 v36; // r8
  __int64 v37; // rcx
  int v38; // r9d
  __int64 v39; // r8
  __int64 *v40; // rdx
  _BYTE *v41; // rsi
  int v42; // ecx
  unsigned int v43; // esi
  unsigned __int64 v44; // r13
  __int64 v45; // r9
  __int64 v46; // rdi
  __int64 *v47; // rcx
  __int64 v48; // r8
  __int64 *v49; // r13
  __int64 *v50; // r14
  __int64 v51; // rdx
  char v52; // r8
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rsi
  unsigned int v56; // edx
  __int64 v57; // rcx
  unsigned int v58; // edi
  unsigned int v59; // r8d
  __int64 *v60; // rax
  __int64 v61; // r10
  __int64 *v62; // r11
  __int64 v63; // r9
  unsigned int v64; // edx
  __int64 *v65; // rax
  __int64 v66; // r8
  __int64 v67; // r8
  __int64 v68; // rax
  _BYTE *v69; // rax
  __int64 v70; // r8
  __int64 v71; // rcx
  int v72; // r9d
  __int64 v73; // r8
  __int64 *v74; // rdx
  _BYTE *v75; // rsi
  int v76; // r14d
  unsigned __int64 *v77; // rax
  int v78; // ecx
  int v79; // ecx
  int v80; // r8d
  int v81; // r8d
  __int64 v82; // r10
  unsigned __int64 *v83; // rsi
  int v84; // edi
  __int64 v85; // rdx
  unsigned __int64 v86; // r9
  int v87; // r9d
  int v88; // r9d
  __int64 v89; // r10
  __int64 v90; // rdx
  unsigned __int64 v91; // r8
  int v92; // edi
  int v93; // edx
  int v94; // edx
  int v95; // r9d
  int v96; // r9d
  int v97; // eax
  int v98; // r10d
  int v99; // eax
  int v100; // r9d
  __int64 v101; // [rsp+8h] [rbp-68h]
  unsigned int v102; // [rsp+10h] [rbp-60h]
  __int64 v103; // [rsp+18h] [rbp-58h]
  _QWORD *v104; // [rsp+18h] [rbp-58h]
  __int64 v105; // [rsp+18h] [rbp-58h]
  __int64 v106; // [rsp+18h] [rbp-58h]
  __int64 v107; // [rsp+18h] [rbp-58h]
  _QWORD *v108; // [rsp+18h] [rbp-58h]
  __int64 v110; // [rsp+20h] [rbp-50h]
  __int64 v111; // [rsp+28h] [rbp-48h]
  __int64 v112[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a1 + 240);
  v6 = *(__int64 **)(*(_QWORD *)(a1 + 232) + 8 * v5 - 8);
  *(_DWORD *)(a1 + 240) = v5 - 1;
  v111 = a1 + 312;
  v7 = *(unsigned __int64 **)(a1 + 320);
  v8 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(unsigned __int64 **)(a1 + 328) != v7 )
    goto LABEL_2;
  v10 = *(unsigned int *)(a1 + 340);
  v11 = &v7[v10];
  if ( v7 != v11 )
  {
    v12 = 0;
    do
    {
      if ( v8 == *v7 )
        goto LABEL_3;
      if ( *v7 == -2 )
        v12 = v7;
      ++v7;
    }
    while ( v11 != v7 );
    if ( v12 )
    {
      *v12 = v8;
      --*(_DWORD *)(a1 + 344);
      ++*(_QWORD *)(a1 + 312);
      if ( (unsigned __int8)sub_1A6F5D0(a1, v6) )
        goto LABEL_4;
      goto LABEL_15;
    }
  }
  if ( (unsigned int)v10 < *(_DWORD *)(a1 + 336) )
  {
    *(_DWORD *)(a1 + 340) = v10 + 1;
    *v11 = v8;
    ++*(_QWORD *)(a1 + 312);
  }
  else
  {
LABEL_2:
    sub_16CCBA0(v111, v8);
  }
LABEL_3:
  if ( (unsigned __int8)sub_1A6F5D0(a1, v6) )
  {
LABEL_4:
    v9 = *(__int64 **)(a1 + 760);
    if ( v9 )
      sub_1A73460(a1, v9, *v6 & 0xFFFFFFFFFFFFFFF8LL, 1);
    *(_QWORD *)(a1 + 760) = v6;
    return;
  }
LABEL_15:
  v13 = sub_1A73F70(a1, 0);
  v14 = v13;
  v15 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_DWORD *)(a1 + 240) || !a2 )
  {
    v110 = sub_1A73B00(a1, v13);
    goto LABEL_18;
  }
  v55 = *(_QWORD *)(a1 + 216);
  v56 = *(_DWORD *)(v55 + 48);
  v57 = *(_QWORD *)(v55 + 32);
  v110 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 32LL);
  if ( !v56 )
  {
LABEL_133:
    *(_BYTE *)(v55 + 72) = 0;
    BUG();
  }
  v58 = v56 - 1;
  v59 = (v56 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v60 = (__int64 *)(v57 + 16LL * v59);
  v61 = *v60;
  if ( v14 == *v60 )
  {
LABEL_59:
    v62 = (__int64 *)(v57 + 16LL * v56);
    if ( v60 != v62 )
    {
      v63 = v60[1];
      goto LABEL_61;
    }
  }
  else
  {
    v99 = 1;
    while ( v61 != -8 )
    {
      v100 = v99 + 1;
      v59 = v58 & (v99 + v59);
      v60 = (__int64 *)(v57 + 16LL * v59);
      v61 = *v60;
      if ( v14 == *v60 )
        goto LABEL_59;
      v99 = v100;
    }
    v62 = (__int64 *)(v57 + 16LL * v56);
  }
  v63 = 0;
LABEL_61:
  v64 = v58 & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
  v65 = (__int64 *)(v57 + 16LL * v64);
  v66 = *v65;
  if ( v110 != *v65 )
  {
    v97 = 1;
    while ( v66 != -8 )
    {
      v98 = v97 + 1;
      v64 = v58 & (v97 + v64);
      v65 = (__int64 *)(v57 + 16LL * v64);
      v66 = *v65;
      if ( v110 == *v65 )
        goto LABEL_62;
      v97 = v98;
    }
    goto LABEL_133;
  }
LABEL_62:
  if ( v62 == v65 )
    goto LABEL_133;
  v67 = v65[1];
  *(_BYTE *)(v55 + 72) = 0;
  v101 = v63;
  v68 = *(_QWORD *)(v67 + 8);
  if ( v68 != v63 )
  {
    v112[0] = v67;
    v69 = sub_1A6CF50(*(_QWORD **)(v68 + 24), *(_QWORD *)(v68 + 32), v112);
    v107 = v70;
    sub_15CDF70(*(_QWORD *)(v70 + 8) + 24LL, v69);
    v72 = v101;
    v73 = v107;
    v74 = v112;
    *(_QWORD *)(v107 + 8) = v101;
    v112[0] = v107;
    v75 = *(_BYTE **)(v101 + 32);
    if ( v75 == *(_BYTE **)(v101 + 40) )
    {
      sub_15CE310(v101 + 24, v75, v112);
      v73 = v107;
    }
    else
    {
      if ( v75 )
      {
        *(_QWORD *)v75 = v107;
        v75 = *(_BYTE **)(v101 + 32);
      }
      v75 += 8;
      *(_QWORD *)(v101 + 32) = v75;
    }
    if ( *(_DWORD *)(v73 + 16) != *(_DWORD *)(*(_QWORD *)(v73 + 8) + 16LL) + 1 )
      sub_1A6CC30(v73, (__int64)v75, (__int64)v74, v71, v73, v72);
  }
  sub_1A73280(a1, v14, v110);
LABEL_18:
  v103 = *(_QWORD *)(a1 + 184);
  v16 = sub_1648A60(56, 3u);
  v18 = v16;
  if ( v16 )
  {
    v19 = v103;
    v104 = v16;
    sub_15F8650((__int64)v16, v15, v110, v19, v14);
    v18 = v104;
  }
  v20 = *(unsigned int *)(a1 + 544);
  if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 548) )
  {
    v108 = v18;
    sub_16CD150(a1 + 536, (const void *)(a1 + 552), 0, 8, v17, (int)v18);
    v20 = *(unsigned int *)(a1 + 544);
    v18 = v108;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 536) + 8 * v20) = v18;
  ++*(_DWORD *)(a1 + 544);
  sub_1A73280(a1, v14, v15);
  v21 = *(_QWORD *)(a1 + 216);
  v22 = *(unsigned int *)(v21 + 48);
  v23 = *(_QWORD *)(v21 + 32);
  if ( !(_DWORD)v22 )
    goto LABEL_131;
  v24 = v22 - 1;
  v25 = (v22 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v26 = (__int64 *)(v23 + 16LL * v25);
  v27 = *v26;
  if ( v14 == *v26 )
  {
LABEL_24:
    v28 = (__int64 *)(v23 + 16 * v22);
    if ( v26 != v28 )
    {
      v29 = v26[1];
      goto LABEL_26;
    }
  }
  else
  {
    v93 = 1;
    while ( v27 != -8 )
    {
      v96 = v93 + 1;
      v25 = v24 & (v93 + v25);
      v26 = (__int64 *)(v23 + 16LL * v25);
      v27 = *v26;
      if ( v14 == *v26 )
        goto LABEL_24;
      v93 = v96;
    }
    v28 = (__int64 *)(v23 + 16 * v22);
  }
  v29 = 0;
LABEL_26:
  v30 = v24 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v31 = (__int64 *)(v23 + 16LL * v30);
  v32 = *v31;
  if ( v15 != *v31 )
  {
    v94 = 1;
    while ( v32 != -8 )
    {
      v95 = v94 + 1;
      v30 = v24 & (v94 + v30);
      v31 = (__int64 *)(v23 + 16LL * v30);
      v32 = *v31;
      if ( v15 == *v31 )
        goto LABEL_27;
      v94 = v95;
    }
LABEL_131:
    *(_BYTE *)(v21 + 72) = 0;
    BUG();
  }
LABEL_27:
  if ( v31 == v28 )
    goto LABEL_131;
  v33 = v31[1];
  *(_BYTE *)(v21 + 72) = 0;
  v34 = *(_QWORD *)(v33 + 8);
  if ( v34 != v29 )
  {
    v112[0] = v33;
    v35 = sub_1A6CF50(*(_QWORD **)(v34 + 24), *(_QWORD *)(v34 + 32), v112);
    v105 = v36;
    sub_15CDF70(*(_QWORD *)(v36 + 8) + 24LL, v35);
    v39 = v105;
    v40 = v112;
    *(_QWORD *)(v105 + 8) = v29;
    v112[0] = v105;
    v41 = *(_BYTE **)(v29 + 32);
    if ( v41 == *(_BYTE **)(v29 + 40) )
    {
      sub_15CE310(v29 + 24, v41, v112);
      v39 = v105;
    }
    else
    {
      if ( v41 )
      {
        *(_QWORD *)v41 = v105;
        v41 = *(_BYTE **)(v29 + 32);
      }
      v41 += 8;
      *(_QWORD *)(v29 + 32) = v41;
    }
    if ( *(_DWORD *)(v39 + 16) != *(_DWORD *)(*(_QWORD *)(v39 + 8) + 16LL) + 1 )
      sub_1A6CC30(v39, (__int64)v41, (__int64)v40, v37, v39, v38);
  }
  v42 = *(_DWORD *)(a1 + 240);
  *(_QWORD *)(a1 + 760) = v6;
  if ( v42 )
  {
    v106 = a1 + 504;
    while ( 1 )
    {
      if ( sub_183E920(v111, a3) )
      {
LABEL_46:
        v6 = *(__int64 **)(a1 + 760);
        goto LABEL_47;
      }
      v43 = *(_DWORD *)(a1 + 528);
      v44 = **(_QWORD **)(*(_QWORD *)(a1 + 232) + 8LL * *(unsigned int *)(a1 + 240) - 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v43 )
        break;
      v45 = *(_QWORD *)(a1 + 512);
      LODWORD(v46) = (v43 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v47 = (__int64 *)(v45 + 40LL * (unsigned int)v46);
      v48 = *v47;
      if ( v44 == *v47 )
      {
LABEL_40:
        v49 = (__int64 *)v47[2];
        v50 = &v49[2 * *((unsigned int *)v47 + 8)];
        if ( *((_DWORD *)v47 + 6) && v49 != v50 )
        {
          while ( 1 )
          {
            v51 = *v49;
            if ( *v49 != -16 && v51 != -8 )
              break;
            v49 += 2;
            if ( v50 == v49 )
              goto LABEL_45;
          }
          while ( v50 != v49 )
          {
            if ( !sub_15CC8F0(*(_QWORD *)(a1 + 216), v15, v51) )
              goto LABEL_46;
            v54 = v49 + 2;
            if ( v50 == v49 + 2 )
              break;
            while ( 1 )
            {
              v51 = *v54;
              v49 = v54;
              if ( *v54 != -8 && v51 != -16 )
                break;
              v54 += 2;
              if ( v50 == v54 )
                goto LABEL_45;
            }
          }
        }
        goto LABEL_45;
      }
      v76 = 1;
      v77 = 0;
      while ( v48 != -8 )
      {
        if ( v48 == -16 && !v77 )
          v77 = (unsigned __int64 *)v47;
        v46 = (v43 - 1) & ((_DWORD)v46 + v76);
        v47 = (__int64 *)(v45 + 40 * v46);
        v48 = *v47;
        if ( v44 == *v47 )
          goto LABEL_40;
        ++v76;
      }
      if ( !v77 )
        v77 = (unsigned __int64 *)v47;
      v78 = *(_DWORD *)(a1 + 520);
      ++*(_QWORD *)(a1 + 504);
      v79 = v78 + 1;
      if ( 4 * v79 >= 3 * v43 )
        goto LABEL_91;
      if ( v43 - *(_DWORD *)(a1 + 524) - v79 > v43 >> 3 )
        goto LABEL_79;
      v102 = ((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4);
      sub_1A6F390(v106, v43);
      v80 = *(_DWORD *)(a1 + 528);
      if ( !v80 )
        goto LABEL_132;
      v81 = v80 - 1;
      v82 = *(_QWORD *)(a1 + 512);
      v83 = 0;
      v84 = 1;
      LODWORD(v85) = v81 & v102;
      v79 = *(_DWORD *)(a1 + 520) + 1;
      v77 = (unsigned __int64 *)(v82 + 40LL * (v81 & v102));
      v86 = *v77;
      if ( v44 == *v77 )
        goto LABEL_79;
      while ( v86 != -8 )
      {
        if ( v86 == -16 && !v83 )
          v83 = v77;
        v85 = v81 & (unsigned int)(v85 + v84);
        v77 = (unsigned __int64 *)(v82 + 40 * v85);
        v86 = *v77;
        if ( v44 == *v77 )
          goto LABEL_79;
        ++v84;
      }
LABEL_95:
      if ( v83 )
        v77 = v83;
LABEL_79:
      *(_DWORD *)(a1 + 520) = v79;
      if ( *v77 != -8 )
        --*(_DWORD *)(a1 + 524);
      *v77 = v44;
      v77[1] = 0;
      v77[2] = 0;
      v77[3] = 0;
      *((_DWORD *)v77 + 8) = 0;
LABEL_45:
      sub_1A74020(a1, 0, a3);
      if ( !*(_DWORD *)(a1 + 240) )
        goto LABEL_46;
    }
    ++*(_QWORD *)(a1 + 504);
LABEL_91:
    sub_1A6F390(v106, 2 * v43);
    v87 = *(_DWORD *)(a1 + 528);
    if ( !v87 )
    {
LABEL_132:
      ++*(_DWORD *)(a1 + 520);
      BUG();
    }
    v88 = v87 - 1;
    v89 = *(_QWORD *)(a1 + 512);
    LODWORD(v90) = v88 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    v79 = *(_DWORD *)(a1 + 520) + 1;
    v77 = (unsigned __int64 *)(v89 + 40LL * (unsigned int)v90);
    v91 = *v77;
    if ( *v77 == v44 )
      goto LABEL_79;
    v92 = 1;
    v83 = 0;
    while ( v91 != -8 )
    {
      if ( v91 == -16 && !v83 )
        v83 = v77;
      v90 = v88 & (unsigned int)(v90 + v92);
      v77 = (unsigned __int64 *)(v89 + 40 * v90);
      v91 = *v77;
      if ( v44 == *v77 )
        goto LABEL_79;
      ++v92;
    }
    goto LABEL_95;
  }
LABEL_47:
  sub_1A73460(a1, v6, v110, 0);
  v52 = sub_1443560(*(_QWORD **)(a1 + 200), v110);
  v53 = 0;
  if ( v52 )
    v53 = sub_1444DB0(*(_QWORD **)(a1 + 200), v110);
  *(_QWORD *)(a1 + 760) = v53;
}
