// Function: sub_2E3AFD0
// Address: 0x2e3afd0
//
__int64 __fastcall sub_2E3AFD0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v8; // r10
  __int64 v9; // r14
  __int64 v10; // rsi
  int v11; // eax
  unsigned int v12; // edx
  _QWORD *v13; // rax
  __int64 v14; // r13
  __int64 v15; // r11
  int v16; // r14d
  __int64 v17; // r15
  int v18; // r11d
  _QWORD *v19; // rdx
  unsigned int v20; // r8d
  _QWORD *v21; // rax
  __int64 v22; // rcx
  int v23; // esi
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // ecx
  int v27; // eax
  __int64 v28; // rdi
  unsigned int v29; // r8d
  _QWORD *v30; // rax
  __int64 v31; // rcx
  int v32; // r14d
  unsigned int v33; // ebx
  const char *v34; // rax
  size_t v35; // rdx
  _BYTE *v36; // rdi
  unsigned __int8 *v37; // rsi
  _BYTE *v38; // rax
  _QWORD *v39; // r8
  __int64 v40; // rax
  _DWORD *v41; // rdx
  __int64 v42; // rdx
  int v44; // eax
  int v45; // ecx
  int v46; // ecx
  __int64 v47; // rdi
  _QWORD *v48; // r8
  unsigned int v49; // r12d
  int v50; // r9d
  __int64 v51; // rsi
  __int64 *v52; // rax
  unsigned __int64 v53; // rdx
  _WORD *v54; // rdx
  const char *v55; // rax
  size_t v56; // rdx
  _BYTE *v57; // rdi
  unsigned __int8 *v58; // rsi
  unsigned __int64 v59; // rax
  _QWORD *v60; // r8
  unsigned __int64 v61; // rax
  int v62; // eax
  int v63; // esi
  __int64 v64; // rdi
  unsigned int v65; // ecx
  int v66; // eax
  __int64 *v67; // rdx
  __int64 v68; // r8
  __int64 v69; // rax
  int v70; // r14d
  int v71; // eax
  int v72; // eax
  int v73; // ecx
  __int64 v74; // rdi
  int v75; // r9d
  unsigned int v76; // r13d
  __int64 *v77; // r8
  __int64 v78; // rsi
  int v79; // r12d
  _QWORD *v80; // r9
  unsigned int v81; // eax
  int v82; // r13d
  _QWORD *v83; // rax
  unsigned __int64 v84; // rax
  unsigned __int64 v85; // rdi
  _QWORD *v86; // rax
  __int64 v87; // rdx
  _QWORD *i; // rdx
  int v89; // r10d
  __int64 *v90; // r9
  __int64 v91; // [rsp+0h] [rbp-B0h]
  __int64 v92; // [rsp+0h] [rbp-B0h]
  __int64 v93; // [rsp+8h] [rbp-A8h]
  __int64 v94; // [rsp+8h] [rbp-A8h]
  size_t v95; // [rsp+8h] [rbp-A8h]
  size_t v96; // [rsp+8h] [rbp-A8h]
  __int64 v97; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v100[2]; // [rsp+20h] [rbp-90h] BYREF
  void (__fastcall *v101)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+30h] [rbp-80h]
  void (__fastcall *v102)(unsigned __int64 *, _QWORD *); // [rsp+38h] [rbp-78h]
  _QWORD v103[3]; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v104; // [rsp+58h] [rbp-58h]
  void *dest; // [rsp+60h] [rbp-50h]
  __int64 v106; // [rsp+68h] [rbp-48h]
  __int64 v107; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)a2 )
  {
    v32 = -1;
    goto LABEL_29;
  }
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(_QWORD **)(a2 + 32);
  v8 = a2 + 24;
  v9 = *(_QWORD *)(a3 + 32);
  v10 = *(unsigned int *)(a2 + 48);
  if ( !v5 )
    goto LABEL_12;
  if ( v5 == v9 )
    goto LABEL_25;
  v11 = *(_DWORD *)(a2 + 40);
  ++*(_QWORD *)(a2 + 24);
  if ( !v11 )
  {
    if ( !*(_DWORD *)(a2 + 44) )
      goto LABEL_12;
    if ( (unsigned int)v10 > 0x40 )
    {
      v94 = v8;
      sub_C7D6A0((__int64)v6, 16 * v10, 8);
      *(_DWORD *)(a2 + 48) = 0;
      LODWORD(v10) = 0;
      v6 = 0;
      *(_QWORD *)(a2 + 32) = 0;
      v15 = v9 + 320;
      v8 = v94;
      *(_QWORD *)(a2 + 40) = 0;
      *(_QWORD *)(a2 + 16) = v9;
      v14 = *(_QWORD *)(v9 + 328);
      if ( v9 + 320 == v14 )
      {
LABEL_79:
        ++*(_QWORD *)(a2 + 24);
        LODWORD(v10) = 0;
        goto LABEL_80;
      }
      goto LABEL_13;
    }
    goto LABEL_8;
  }
  v12 = 4 * v11;
  if ( (unsigned int)(4 * v11) < 0x40 )
    v12 = 64;
  if ( (unsigned int)v10 <= v12 )
  {
LABEL_8:
    v13 = &v6[2 * (unsigned int)v10];
    if ( v13 != v6 )
    {
      do
      {
        *v6 = -4096;
        v6 += 2;
      }
      while ( v13 != v6 );
      v6 = *(_QWORD **)(a2 + 32);
      LODWORD(v10) = *(_DWORD *)(a2 + 48);
    }
    *(_QWORD *)(a2 + 40) = 0;
    goto LABEL_12;
  }
  v81 = v11 - 1;
  if ( v81 )
  {
    _BitScanReverse(&v81, v81);
    v82 = 1 << (33 - (v81 ^ 0x1F));
    if ( v82 < 64 )
      v82 = 64;
    if ( v82 == (_DWORD)v10 )
    {
      *(_QWORD *)(a2 + 40) = 0;
      v83 = &v6[2 * (unsigned int)v82];
      do
      {
        if ( v6 )
          *v6 = -4096;
        v6 += 2;
      }
      while ( v83 != v6 );
      v6 = *(_QWORD **)(a2 + 32);
      LODWORD(v10) = *(_DWORD *)(a2 + 48);
      goto LABEL_12;
    }
  }
  else
  {
    v82 = 64;
  }
  v97 = v8;
  sub_C7D6A0((__int64)v6, 16 * v10, 8);
  v84 = ((((((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
           | (4 * v82 / 3u + 1)
           | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
         | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
         | (4 * v82 / 3u + 1)
         | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
         | (4 * v82 / 3u + 1)
         | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
       | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
       | (4 * v82 / 3u + 1)
       | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 16;
  v85 = (v84
       | (((((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
           | (4 * v82 / 3u + 1)
           | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
         | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
         | (4 * v82 / 3u + 1)
         | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
         | (4 * v82 / 3u + 1)
         | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 4)
       | (((4 * v82 / 3u + 1) | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1)) >> 2)
       | (4 * v82 / 3u + 1)
       | ((unsigned __int64)(4 * v82 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a2 + 48) = v85;
  v86 = (_QWORD *)sub_C7D670(16 * v85, 8);
  v87 = *(unsigned int *)(a2 + 48);
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 32) = v86;
  v8 = v97;
  v6 = v86;
  LODWORD(v10) = v87;
  for ( i = &v86[2 * v87]; i != v86; v86 += 2 )
  {
    if ( v86 )
      *v86 = -4096;
  }
LABEL_12:
  *(_QWORD *)(a2 + 16) = v9;
  v14 = *(_QWORD *)(v9 + 328);
  v15 = v9 + 320;
  if ( v14 == v9 + 320 )
    goto LABEL_25;
LABEL_13:
  v93 = a1;
  v16 = 0;
  v17 = v15;
  while ( 1 )
  {
    if ( !(_DWORD)v10 )
    {
      ++*(_QWORD *)(a2 + 24);
LABEL_19:
      v91 = v8;
      sub_2E3ADF0(v8, 2 * v10);
      v23 = *(_DWORD *)(a2 + 48);
      if ( !v23 )
        goto LABEL_151;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a2 + 32);
      v8 = v91;
      v26 = v24 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v27 = *(_DWORD *)(a2 + 40) + 1;
      v19 = (_QWORD *)(v25 + 16LL * v26);
      v28 = *v19;
      if ( v14 != *v19 )
      {
        v79 = 1;
        v80 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v80 )
            v80 = v19;
          v26 = v24 & (v79 + v26);
          v19 = (_QWORD *)(v25 + 16LL * v26);
          v28 = *v19;
          if ( v14 == *v19 )
            goto LABEL_21;
          ++v79;
        }
        if ( v80 )
          v19 = v80;
      }
      goto LABEL_21;
    }
    v18 = 1;
    v19 = 0;
    v20 = (v10 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v21 = &v6[2 * v20];
    v22 = *v21;
    if ( v14 != *v21 )
      break;
LABEL_15:
    *((_DWORD *)v21 + 2) = v16;
    v14 = *(_QWORD *)(v14 + 8);
    ++v16;
    if ( v14 == v17 )
      goto LABEL_24;
LABEL_16:
    v6 = *(_QWORD **)(a2 + 32);
    LODWORD(v10) = *(_DWORD *)(a2 + 48);
  }
  while ( v22 != -4096 )
  {
    if ( !v19 && v22 == -8192 )
      v19 = v21;
    v20 = (v10 - 1) & (v18 + v20);
    v21 = &v6[2 * v20];
    v22 = *v21;
    if ( v14 == *v21 )
      goto LABEL_15;
    ++v18;
  }
  if ( !v19 )
    v19 = v21;
  v44 = *(_DWORD *)(a2 + 40);
  ++*(_QWORD *)(a2 + 24);
  v27 = v44 + 1;
  if ( 4 * v27 >= (unsigned int)(3 * v10) )
    goto LABEL_19;
  if ( (int)v10 - (v27 + *(_DWORD *)(a2 + 44)) <= (unsigned int)v10 >> 3 )
  {
    v92 = v8;
    sub_2E3ADF0(v8, v10);
    v45 = *(_DWORD *)(a2 + 48);
    if ( !v45 )
      goto LABEL_151;
    v46 = v45 - 1;
    v47 = *(_QWORD *)(a2 + 32);
    v48 = 0;
    v49 = v46 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v8 = v92;
    v50 = 1;
    v27 = *(_DWORD *)(a2 + 40) + 1;
    v19 = (_QWORD *)(v47 + 16LL * v49);
    v51 = *v19;
    if ( v14 != *v19 )
    {
      while ( v51 != -4096 )
      {
        if ( !v48 && v51 == -8192 )
          v48 = v19;
        v49 = v46 & (v50 + v49);
        v19 = (_QWORD *)(v47 + 16LL * v49);
        v51 = *v19;
        if ( v14 == *v19 )
          goto LABEL_21;
        ++v50;
      }
      if ( v48 )
        v19 = v48;
    }
  }
LABEL_21:
  *(_DWORD *)(a2 + 40) = v27;
  if ( *v19 != -4096 )
    --*(_DWORD *)(a2 + 44);
  *v19 = v14;
  *((_DWORD *)v19 + 2) = 0;
  *((_DWORD *)v19 + 2) = v16;
  v14 = *(_QWORD *)(v14 + 8);
  ++v16;
  if ( v14 != v17 )
    goto LABEL_16;
LABEL_24:
  a1 = v93;
  v6 = *(_QWORD **)(a2 + 32);
  LODWORD(v10) = *(_DWORD *)(a2 + 48);
LABEL_25:
  if ( !(_DWORD)v10 )
    goto LABEL_79;
  v29 = (v10 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v30 = &v6[2 * v29];
  v31 = *v30;
  if ( *v30 == a3 )
  {
LABEL_27:
    v32 = *((_DWORD *)v30 + 2);
    goto LABEL_29;
  }
  v70 = 1;
  v67 = 0;
  while ( v31 != -4096 )
  {
    if ( !v67 && v31 == -8192 )
      v67 = v30;
    v29 = (v10 - 1) & (v70 + v29);
    v30 = &v6[2 * v29];
    v31 = *v30;
    if ( *v30 == a3 )
      goto LABEL_27;
    ++v70;
  }
  if ( !v67 )
    v67 = v30;
  v71 = *(_DWORD *)(a2 + 40);
  ++*(_QWORD *)(a2 + 24);
  v66 = v71 + 1;
  if ( 4 * v66 < (unsigned int)(3 * v10) )
  {
    if ( (int)v10 - (v66 + *(_DWORD *)(a2 + 44)) > (unsigned int)v10 >> 3 )
      goto LABEL_82;
    sub_2E3ADF0(v8, v10);
    v72 = *(_DWORD *)(a2 + 48);
    if ( v72 )
    {
      v73 = v72 - 1;
      v74 = *(_QWORD *)(a2 + 32);
      v75 = 1;
      v76 = (v72 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v77 = 0;
      v66 = *(_DWORD *)(a2 + 40) + 1;
      v67 = (__int64 *)(v74 + 16LL * v76);
      v78 = *v67;
      if ( *v67 != a3 )
      {
        while ( v78 != -4096 )
        {
          if ( v78 == -8192 && !v77 )
            v77 = v67;
          v76 = v73 & (v75 + v76);
          v67 = (__int64 *)(v74 + 16LL * v76);
          v78 = *v67;
          if ( *v67 == a3 )
            goto LABEL_82;
          ++v75;
        }
        if ( v77 )
          v67 = v77;
      }
      goto LABEL_82;
    }
LABEL_151:
    ++*(_DWORD *)(a2 + 40);
LABEL_152:
    BUG();
  }
LABEL_80:
  sub_2E3ADF0(v8, 2 * v10);
  v62 = *(_DWORD *)(a2 + 48);
  if ( !v62 )
    goto LABEL_151;
  v63 = v62 - 1;
  v64 = *(_QWORD *)(a2 + 32);
  v65 = (v62 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v66 = *(_DWORD *)(a2 + 40) + 1;
  v67 = (__int64 *)(v64 + 16LL * v65);
  v68 = *v67;
  if ( *v67 != a3 )
  {
    v89 = 1;
    v90 = 0;
    while ( v68 != -4096 )
    {
      if ( !v90 && v68 == -8192 )
        v90 = v67;
      v65 = v63 & (v89 + v65);
      v67 = (__int64 *)(v64 + 16LL * v65);
      v68 = *v67;
      if ( *v67 == a3 )
        goto LABEL_82;
      ++v89;
    }
    if ( v90 )
      v67 = v90;
  }
LABEL_82:
  *(_DWORD *)(a2 + 40) = v66;
  if ( *v67 != -4096 )
    --*(_DWORD *)(a2 + 44);
  *((_DWORD *)v67 + 2) = 0;
  v32 = 0;
  *v67 = a3;
LABEL_29:
  v103[1] = 0;
  v103[2] = 0;
  v33 = qword_501ED48[8];
  v104 = 0;
  dest = 0;
  if ( !LODWORD(qword_501ED48[8]) )
    v33 = qword_501EFE8;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v106 = 0x100000000LL;
  v103[0] = &unk_49DD210;
  v107 = a1;
  sub_CB5980((__int64)v103, 0, 0, 0);
  if ( v32 == -1 )
  {
    v55 = sub_2E31BC0(a3);
    v57 = dest;
    v58 = (unsigned __int8 *)v55;
    v59 = v104 - (_BYTE *)dest;
    if ( v104 - (_BYTE *)dest < v56 )
    {
      v69 = sub_CB6200((__int64)v103, v58, v56);
      v57 = *(_BYTE **)(v69 + 32);
      v60 = (_QWORD *)v69;
      v59 = *(_QWORD *)(v69 + 24) - (_QWORD)v57;
    }
    else
    {
      v60 = v103;
      if ( v56 )
      {
        v95 = v56;
        memcpy(dest, v58, v56);
        v60 = v103;
        dest = (char *)dest + v95;
        v57 = dest;
        v59 = v104 - (_BYTE *)dest;
      }
    }
    if ( v59 <= 2 )
    {
      sub_CB6200((__int64)v60, (unsigned __int8 *)" : ", 3u);
    }
    else
    {
      v57[2] = 32;
      *(_WORD *)v57 = 14880;
      v60[4] += 3LL;
    }
    goto LABEL_39;
  }
  v34 = sub_2E31BC0(a3);
  v36 = dest;
  v37 = (unsigned __int8 *)v34;
  v38 = v104;
  if ( v104 - (_BYTE *)dest < v35 )
  {
    v39 = (_QWORD *)sub_CB6200((__int64)v103, v37, v35);
    v36 = (_BYTE *)v39[4];
    if ( v36 != (_BYTE *)v39[3] )
      goto LABEL_36;
    goto LABEL_47;
  }
  v39 = v103;
  if ( v35 )
  {
    v96 = v35;
    memcpy(dest, v37, v35);
    dest = (char *)dest + v96;
    v38 = v104;
    v36 = dest;
    v39 = v103;
  }
  if ( v36 == v38 )
  {
LABEL_47:
    v39 = (_QWORD *)sub_CB6200((__int64)v39, (unsigned __int8 *)"[", 1u);
    goto LABEL_37;
  }
LABEL_36:
  *v36 = 91;
  ++v39[4];
LABEL_37:
  v40 = sub_CB59F0((__int64)v39, v32);
  v41 = *(_DWORD **)(v40 + 32);
  if ( *(_QWORD *)(v40 + 24) - (_QWORD)v41 <= 3u )
  {
    sub_CB6200(v40, "] : ", 4u);
  }
  else
  {
    *v41 = 540680285;
    *(_QWORD *)(v40 + 32) += 4LL;
  }
LABEL_39:
  if ( v33 == 2 )
  {
    v61 = sub_2E39EA0(a4, a3);
    sub_CB59D0((__int64)v103, v61);
  }
  else if ( v33 > 2 )
  {
    if ( v33 == 3 )
    {
      v52 = sub_2E39F50(a4, a3);
      v100[1] = v53;
      v100[0] = (unsigned __int64)v52;
      if ( (_BYTE)v53 )
      {
        sub_CB59D0((__int64)v103, v100[0]);
      }
      else
      {
        v54 = dest;
        if ( (unsigned __int64)(v104 - (_BYTE *)dest) <= 6 )
        {
          sub_CB6200((__int64)v103, "Unknown", 7u);
        }
        else
        {
          *(_DWORD *)dest = 1852534357;
          v54[2] = 30575;
          *((_BYTE *)v54 + 6) = 110;
          dest = (char *)dest + 7;
        }
      }
    }
  }
  else
  {
    if ( !v33 )
      goto LABEL_152;
    sub_2E3A100(v100, a4, a3);
    if ( !v101 )
      sub_4263D6(v100, a4, v42);
    v102(v100, v103);
    if ( v101 )
      v101(v100, v100, 3);
  }
  v103[0] = &unk_49DD210;
  sub_CB5840((__int64)v103);
  return a1;
}
