// Function: sub_1A74020
// Address: 0x1a74020
//
__int64 __fastcall sub_1A74020(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  int v6; // edx
  __int64 v7; // rcx
  __int64 v8; // rax
  int v9; // r9d
  __int64 v10; // rdi
  __int64 *v11; // r14
  int v12; // ecx
  unsigned __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // edx
  __int64 v16; // rsi
  unsigned int v17; // esi
  unsigned __int64 v18; // r14
  __int64 v19; // r8
  unsigned int v20; // edi
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // r15
  _QWORD *v27; // rax
  int v28; // r8d
  int v29; // r9d
  _QWORD *v30; // rbx
  __int64 v31; // rax
  char v32; // r8
  __int64 result; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // r8d
  unsigned int v38; // ecx
  __int64 *v39; // rdx
  __int64 v40; // r9
  __int64 *v41; // rax
  __int64 v42; // rcx
  unsigned int v43; // r9d
  __int64 *v44; // rdx
  __int64 v45; // r11
  __int64 v46; // rbx
  __int64 v47; // rax
  _BYTE *v48; // rax
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // rcx
  __int64 *v52; // rdx
  _BYTE *v53; // rsi
  __int64 v54; // r15
  _QWORD *v55; // rax
  __int64 v56; // r14
  _QWORD *v57; // rdi
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 *v60; // rax
  int v61; // r9d
  __int64 v62; // r8
  __int64 v63; // r10
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r14
  __int64 v70; // rsi
  __int64 v71; // r9
  __int64 v72; // rdi
  __int64 v73; // rdi
  int v74; // eax
  int v75; // esi
  __int64 v76; // rdi
  unsigned int v77; // eax
  int v78; // ecx
  unsigned __int64 *v79; // rdx
  unsigned __int64 v80; // r8
  __int64 v81; // r14
  int v82; // r11d
  int v83; // eax
  int v84; // eax
  int v85; // eax
  int v86; // r9d
  unsigned __int64 *v87; // r8
  __int64 v88; // rdi
  unsigned int v89; // r15d
  unsigned __int64 v90; // rsi
  int v91; // edx
  int v92; // r10d
  int v93; // edx
  int v94; // edx
  unsigned __int64 *v95; // r11
  int v96; // r10d
  unsigned __int64 *v97; // r9
  int v98; // r10d
  __int64 v99; // [rsp+0h] [rbp-80h]
  __int64 *v100; // [rsp+8h] [rbp-78h]
  __int64 v101; // [rsp+8h] [rbp-78h]
  __int64 v102; // [rsp+10h] [rbp-70h]
  __int64 v103; // [rsp+10h] [rbp-70h]
  __int64 v104; // [rsp+10h] [rbp-70h]
  __int64 *v105; // [rsp+10h] [rbp-70h]
  __int64 *v106; // [rsp+10h] [rbp-70h]
  __int64 v107; // [rsp+10h] [rbp-70h]
  __int64 v108; // [rsp+10h] [rbp-70h]
  __int64 v109; // [rsp+18h] [rbp-68h]
  __int64 v110; // [rsp+18h] [rbp-68h]
  __int64 v111; // [rsp+18h] [rbp-68h]
  __int64 v112; // [rsp+18h] [rbp-68h]
  __int64 v113; // [rsp+18h] [rbp-68h]
  __int64 v114; // [rsp+18h] [rbp-68h]
  __int64 v115; // [rsp+20h] [rbp-60h] BYREF
  __int64 v116; // [rsp+28h] [rbp-58h] BYREF
  __int64 v117[2]; // [rsp+30h] [rbp-50h] BYREF
  char v118; // [rsp+40h] [rbp-40h]
  char v119; // [rsp+41h] [rbp-3Fh]

  v6 = *(_DWORD *)(a1 + 640);
  if ( !v6 )
    return sub_1A74980(a1, a2, a3);
  v7 = *(unsigned int *)(a1 + 240);
  v8 = *(_QWORD *)(a1 + 232);
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 624);
  v11 = *(__int64 **)(v8 + 8 * v7 - 8);
  v12 = v6 - 1;
  v13 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = v13;
  v15 = (v6 - 1) & (((*(_DWORD *)v11 & 0xFFFFFFF8) >> 9) ^ ((unsigned int)v13 >> 4));
  v16 = *(_QWORD *)(v10 + 16LL * (v12 & (((*(_DWORD *)v11 & 0xFFFFFFF8) >> 9) ^ ((unsigned int)v13 >> 4))));
  if ( v13 != v16 )
  {
    while ( v16 != -8 )
    {
      v15 = v12 & (v9 + v15);
      v16 = *(_QWORD *)(v10 + 16LL * v15);
      if ( v13 == v16 )
        goto LABEL_3;
      ++v9;
    }
    return sub_1A74980(a1, a2, a3);
  }
LABEL_3:
  if ( !(unsigned __int8)sub_1A6F5D0(a1, v11) )
    v14 = sub_1A73F70(a1, 1);
  v17 = *(_DWORD *)(a1 + 640);
  v18 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v17 )
  {
    ++*(_QWORD *)(a1 + 616);
    goto LABEL_63;
  }
  v19 = *(_QWORD *)(a1 + 624);
  v20 = (v17 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v21 = (unsigned __int64 *)(v19 + 16LL * v20);
  v22 = *v21;
  if ( v18 == *v21 )
  {
    v23 = v21[1];
    goto LABEL_8;
  }
  v82 = 1;
  v79 = 0;
  while ( v22 != -8 )
  {
    if ( v79 || v22 != -16 )
      v21 = v79;
    v94 = v82 + 1;
    v20 = (v17 - 1) & (v82 + v20);
    v95 = (unsigned __int64 *)(v19 + 16LL * v20);
    v22 = *v95;
    if ( v18 == *v95 )
    {
      v23 = v95[1];
      goto LABEL_8;
    }
    v82 = v94;
    v79 = v21;
    v21 = (unsigned __int64 *)(v19 + 16LL * v20);
  }
  if ( !v79 )
    v79 = v21;
  v83 = *(_DWORD *)(a1 + 632);
  ++*(_QWORD *)(a1 + 616);
  v78 = v83 + 1;
  if ( 4 * (v83 + 1) >= 3 * v17 )
  {
LABEL_63:
    sub_1447B20(a1 + 616, 2 * v17);
    v74 = *(_DWORD *)(a1 + 640);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a1 + 624);
      v77 = (v74 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v78 = *(_DWORD *)(a1 + 632) + 1;
      v79 = (unsigned __int64 *)(v76 + 16LL * v77);
      v80 = *v79;
      if ( v18 != *v79 )
      {
        v96 = 1;
        v97 = 0;
        while ( v80 != -8 )
        {
          if ( v80 == -16 && !v97 )
            v97 = v79;
          v77 = v75 & (v96 + v77);
          v79 = (unsigned __int64 *)(v76 + 16LL * v77);
          v80 = *v79;
          if ( v18 == *v79 )
            goto LABEL_65;
          ++v96;
        }
        if ( v97 )
          v79 = v97;
      }
      goto LABEL_65;
    }
    goto LABEL_117;
  }
  if ( v17 - *(_DWORD *)(a1 + 636) - v78 <= v17 >> 3 )
  {
    sub_1447B20(a1 + 616, v17);
    v84 = *(_DWORD *)(a1 + 640);
    if ( v84 )
    {
      v85 = v84 - 1;
      v86 = 1;
      v87 = 0;
      v88 = *(_QWORD *)(a1 + 624);
      v89 = v85 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v78 = *(_DWORD *)(a1 + 632) + 1;
      v79 = (unsigned __int64 *)(v88 + 16LL * v89);
      v90 = *v79;
      if ( v18 != *v79 )
      {
        while ( v90 != -8 )
        {
          if ( v90 == -16 && !v87 )
            v87 = v79;
          v89 = v85 & (v86 + v89);
          v79 = (unsigned __int64 *)(v88 + 16LL * v89);
          v90 = *v79;
          if ( v18 == *v79 )
            goto LABEL_65;
          ++v86;
        }
        if ( v87 )
          v79 = v87;
      }
      goto LABEL_65;
    }
LABEL_117:
    ++*(_DWORD *)(a1 + 632);
    BUG();
  }
LABEL_65:
  *(_DWORD *)(a1 + 632) = v78;
  if ( *v79 != -8 )
    --*(_DWORD *)(a1 + 636);
  *v79 = v18;
  v23 = 0;
  v79[1] = 0;
LABEL_8:
  sub_1A74980(a1, 0, v23);
  while ( !sub_183E920(a1 + 312, v23) )
    sub_1A74020(a1, 0, v23);
  v24 = *(_QWORD *)(*(_QWORD *)(v14 + 56) + 80LL);
  if ( v24 && v14 == v24 - 24 )
  {
    v102 = *(_QWORD *)(v14 + 56);
    v117[0] = (__int64)"entry.orig";
    v119 = 1;
    v118 = 3;
    sub_164B780(v14, v117);
    v119 = 1;
    v117[0] = (__int64)"entry";
    v118 = 3;
    v54 = sub_157E9C0(v14);
    v55 = (_QWORD *)sub_22077B0(64);
    v56 = (__int64)v55;
    if ( v55 )
      sub_157FB60(v55, v54, (__int64)v117, v102, v14);
    v57 = sub_1648A60(56, 1u);
    if ( v57 )
      sub_15F8590((__int64)v57, v14, v56);
    v58 = *(_QWORD *)(a1 + 216);
    v115 = v56;
    *(_BYTE *)(v58 + 72) = 0;
    v59 = sub_22077B0(56);
    if ( v59 )
    {
      *(_QWORD *)v59 = v56;
      *(_QWORD *)(v59 + 8) = 0;
      *(_DWORD *)(v59 + 16) = 0;
      *(_QWORD *)(v59 + 24) = 0;
      *(_QWORD *)(v59 + 32) = 0;
      *(_QWORD *)(v59 + 40) = 0;
      *(_QWORD *)(v59 + 48) = -1;
    }
    v103 = v59;
    v60 = sub_15CFF10(v58 + 24, &v115);
    v62 = v103;
    v63 = v60[1];
    v60[1] = v103;
    if ( v63 )
    {
      v64 = *(_QWORD *)(v63 + 24);
      if ( v64 )
      {
        v100 = v60;
        v104 = v63;
        j_j___libc_free_0(v64, *(_QWORD *)(v63 + 40) - v64);
        v60 = v100;
        v63 = v104;
      }
      v105 = v60;
      j_j___libc_free_0(v63, 56);
      v62 = v105[1];
    }
    v65 = *(unsigned int *)(v58 + 8);
    if ( (_DWORD)v65 )
    {
      v99 = v62;
      v116 = **(_QWORD **)v58;
      v106 = sub_15CFF10(v58 + 24, &v116);
      v66 = sub_15CFF10(v58 + 24, &v116);
      v62 = v99;
      v68 = (__int64)v106;
      v69 = v66[1];
      v66[1] = 0;
      v117[0] = v69;
      v70 = *(_QWORD *)(v99 + 32);
      if ( v70 == *(_QWORD *)(v99 + 40) )
      {
        sub_15CE310(v99 + 24, (_BYTE *)v70, v117);
        v68 = (__int64)v106;
        v62 = v99;
      }
      else
      {
        if ( v70 )
        {
          *(_QWORD *)v70 = v69;
          v70 = *(_QWORD *)(v99 + 32);
        }
        v70 += 8;
        *(_QWORD *)(v99 + 32) = v70;
      }
      v71 = *(_QWORD *)(v68 + 8);
      *(_QWORD *)(v68 + 8) = v69;
      if ( v71 )
      {
        v72 = *(_QWORD *)(v71 + 24);
        if ( v72 )
        {
          v101 = v68;
          v107 = v62;
          v111 = v71;
          j_j___libc_free_0(v72, *(_QWORD *)(v71 + 40) - v72);
          v68 = v101;
          v62 = v107;
          v71 = v111;
        }
        v70 = 56;
        v108 = v68;
        v112 = v62;
        j_j___libc_free_0(v71, 56);
        v68 = v108;
        v62 = v112;
        v69 = *(_QWORD *)(v108 + 8);
      }
      *(_QWORD *)(v69 + 8) = v62;
      v73 = *(_QWORD *)(v68 + 8);
      if ( *(_DWORD *)(v73 + 16) != *(_DWORD *)(*(_QWORD *)(v73 + 8) + 16LL) + 1 )
      {
        v113 = v62;
        sub_1A6CC30(v73, v70, v67, v68, v62, v71);
        v62 = v113;
      }
      **(_QWORD **)v58 = v115;
    }
    else
    {
      v81 = v115;
      if ( !*(_DWORD *)(v58 + 12) )
      {
        v114 = v62;
        sub_16CD150(v58, (const void *)(v58 + 16), 0, 8, v62, v61);
        v65 = *(unsigned int *)(v58 + 8);
        v62 = v114;
      }
      *(_QWORD *)(*(_QWORD *)v58 + 8 * v65) = v81;
      ++*(_DWORD *)(v58 + 8);
    }
    *(_QWORD *)(v58 + 56) = v62;
  }
  v25 = sub_1A73F70(a1, 0);
  if ( !*(_DWORD *)(a1 + 240) && a2 )
  {
    v34 = *(_QWORD *)(a1 + 216);
    v26 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 32LL);
    v35 = *(unsigned int *)(v34 + 48);
    v36 = *(_QWORD *)(v34 + 32);
    if ( !(_DWORD)v35 )
      goto LABEL_118;
    v37 = v35 - 1;
    v38 = (v35 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v39 = (__int64 *)(v36 + 16LL * v38);
    v40 = *v39;
    if ( v25 == *v39 )
    {
LABEL_28:
      v41 = (__int64 *)(v36 + 16 * v35);
      if ( v41 != v39 )
      {
        v42 = v39[1];
        goto LABEL_30;
      }
    }
    else
    {
      v93 = 1;
      while ( v40 != -8 )
      {
        v98 = v93 + 1;
        v38 = v37 & (v93 + v38);
        v39 = (__int64 *)(v36 + 16LL * v38);
        v40 = *v39;
        if ( v25 == *v39 )
          goto LABEL_28;
        v93 = v98;
      }
      v41 = (__int64 *)(v36 + 16 * v35);
    }
    v42 = 0;
LABEL_30:
    v43 = v37 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v44 = (__int64 *)(v36 + 16LL * v43);
    v45 = *v44;
    if ( v26 == *v44 )
    {
LABEL_31:
      if ( v41 != v44 )
      {
        v46 = v44[1];
        *(_BYTE *)(v34 + 72) = 0;
        v110 = v42;
        v47 = *(_QWORD *)(v46 + 8);
        if ( v42 != v47 )
        {
          v117[0] = v46;
          v48 = sub_1A6CF50(*(_QWORD **)(v47 + 24), *(_QWORD *)(v47 + 32), v117);
          sub_15CDF70(*(_QWORD *)(v46 + 8) + 24LL, v48);
          v51 = v110;
          v52 = v117;
          *(_QWORD *)(v46 + 8) = v110;
          v117[0] = v46;
          v53 = *(_BYTE **)(v110 + 32);
          if ( v53 == *(_BYTE **)(v110 + 40) )
          {
            sub_15CE310(v110 + 24, v53, v117);
          }
          else
          {
            if ( v53 )
            {
              *(_QWORD *)v53 = v46;
              v53 = *(_BYTE **)(v110 + 32);
            }
            v53 += 8;
            *(_QWORD *)(v110 + 32) = v53;
          }
          if ( *(_DWORD *)(v46 + 16) != *(_DWORD *)(*(_QWORD *)(v46 + 8) + 16LL) + 1 )
            sub_1A6CC30(v46, (__int64)v53, (__int64)v52, v51, v49, v50);
        }
        sub_1A73280(a1, v25, v26);
        goto LABEL_15;
      }
    }
    else
    {
      v91 = 1;
      while ( v45 != -8 )
      {
        v92 = v91 + 1;
        v43 = v37 & (v91 + v43);
        v44 = (__int64 *)(v36 + 16LL * v43);
        v45 = *v44;
        if ( v26 == *v44 )
          goto LABEL_31;
        v91 = v92;
      }
    }
LABEL_118:
    *(_BYTE *)(v34 + 72) = 0;
    BUG();
  }
  v26 = sub_1A73B00(a1, v25);
LABEL_15:
  v109 = *(_QWORD *)(a1 + 184);
  v27 = sub_1648A60(56, 3u);
  v30 = v27;
  if ( v27 )
    sub_15F8650((__int64)v27, v26, v14, v109, v25);
  v31 = *(unsigned int *)(a1 + 688);
  if ( (unsigned int)v31 >= *(_DWORD *)(a1 + 692) )
  {
    sub_16CD150(a1 + 680, (const void *)(a1 + 696), 0, 8, v28, v29);
    v31 = *(unsigned int *)(a1 + 688);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 680) + 8 * v31) = v30;
  ++*(_DWORD *)(a1 + 688);
  sub_1A73280(a1, v25, v14);
  v32 = sub_1443560(*(_QWORD **)(a1 + 200), v26);
  result = 0;
  if ( v32 )
    result = sub_1444DB0(*(_QWORD **)(a1 + 200), v26);
  *(_QWORD *)(a1 + 760) = result;
  return result;
}
