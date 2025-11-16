// Function: sub_356FB10
// Address: 0x356fb10
//
__int64 __fastcall sub_356FB10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v6; // r12
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r14d
  __int64 *v11; // rdx
  unsigned int v12; // edi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rbx
  unsigned int *v19; // r11
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  unsigned int v23; // r12d
  __int64 v24; // r15
  __int64 v25; // rax
  unsigned int v26; // esi
  int v27; // r8d
  __int64 v28; // r13
  __int64 v29; // r9
  unsigned int v30; // edi
  _QWORD *v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // r13
  int v35; // eax
  unsigned int *v36; // r10
  __int64 *v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  int v44; // eax
  int v45; // eax
  _QWORD *v46; // rdx
  int v47; // eax
  int v48; // eax
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned __int64 v51; // rcx
  int v52; // esi
  int v53; // esi
  __int64 v54; // r9
  unsigned int v55; // ecx
  __int64 v56; // rdi
  int v57; // r14d
  _QWORD *v58; // r10
  int v59; // ecx
  int v60; // ecx
  __int64 v61; // rdi
  _QWORD *v62; // r9
  unsigned int v63; // r14d
  int v64; // r10d
  __int64 v65; // rsi
  int v66; // ecx
  int v67; // ecx
  __int64 v68; // rsi
  __int64 v69; // rdi
  int v70; // r13d
  int v71; // ecx
  int v72; // ecx
  __int64 v73; // rdi
  unsigned int v74; // r13d
  __int64 v75; // rsi
  unsigned int v76; // r14d
  unsigned int *v77; // [rsp+8h] [rbp-108h]
  unsigned int *v78; // [rsp+8h] [rbp-108h]
  unsigned int *v79; // [rsp+8h] [rbp-108h]
  unsigned __int64 v80; // [rsp+8h] [rbp-108h]
  unsigned int *v81; // [rsp+10h] [rbp-100h]
  __int64 v82; // [rsp+10h] [rbp-100h]
  int v83; // [rsp+10h] [rbp-100h]
  int v84; // [rsp+10h] [rbp-100h]
  int v85; // [rsp+10h] [rbp-100h]
  __int64 v86; // [rsp+10h] [rbp-100h]
  __int64 v87; // [rsp+10h] [rbp-100h]
  unsigned __int64 v88; // [rsp+10h] [rbp-100h]
  __int64 v89; // [rsp+18h] [rbp-F8h]
  int v90; // [rsp+20h] [rbp-F0h]
  __int64 v91; // [rsp+20h] [rbp-F0h]
  __int64 v92; // [rsp+20h] [rbp-F0h]
  __int64 v93; // [rsp+20h] [rbp-F0h]
  unsigned int *v94; // [rsp+28h] [rbp-E8h]
  _QWORD *v95; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v96; // [rsp+38h] [rbp-D8h]
  _QWORD v97[26]; // [rsp+40h] [rbp-D0h] BYREF

  v3 = a1;
  v6 = *(_QWORD *)(a2 + 24);
  v95 = v97;
  v97[0] = a2;
  v7 = *(_DWORD *)(a1 + 48);
  v96 = 0x1400000001LL;
  v89 = a1 + 24;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 32);
    v9 = v7 - 1;
    v10 = 1;
    v11 = 0;
    v12 = v9 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v13 = (__int64 *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( v6 == *v13 )
    {
LABEL_3:
      v15 = v13[1];
      v16 = v15;
      goto LABEL_4;
    }
    while ( v14 != -4096 )
    {
      if ( !v11 && v14 == -8192 )
        v11 = v13;
      v12 = v9 & (v10 + v12);
      v13 = (__int64 *)(v8 + 16LL * v12);
      v14 = *v13;
      if ( v6 == *v13 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v11 )
      v11 = v13;
    v44 = *(_DWORD *)(v3 + 40);
    ++*(_QWORD *)(v3 + 24);
    v45 = v44 + 1;
    if ( 4 * v45 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(v3 + 44) - v45 <= v7 >> 3 )
      {
        v93 = v3;
        v87 = a3;
        sub_356EA90(v89, v7);
        v3 = v93;
        v71 = *(_DWORD *)(v93 + 48);
        if ( !v71 )
          goto LABEL_111;
        v72 = v71 - 1;
        v73 = *(_QWORD *)(v93 + 32);
        v8 = 0;
        v74 = v72 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        a3 = v87;
        v9 = 1;
        v45 = *(_DWORD *)(v93 + 40) + 1;
        v11 = (__int64 *)(v73 + 16LL * v74);
        v75 = *v11;
        if ( v6 != *v11 )
        {
          while ( v75 != -4096 )
          {
            if ( v75 == -8192 && !v8 )
              v8 = (__int64)v11;
            v76 = v9 + 1;
            v9 = v72 & (v74 + (unsigned int)v9);
            v74 = v9;
            v11 = (__int64 *)(v73 + 16LL * (unsigned int)v9);
            v75 = *v11;
            if ( v6 == *v11 )
              goto LABEL_40;
            v9 = v76;
          }
          if ( v8 )
            v11 = (__int64 *)v8;
        }
      }
      goto LABEL_40;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  v92 = v3;
  v86 = a3;
  sub_356EA90(v89, 2 * v7);
  v3 = v92;
  v66 = *(_DWORD *)(v92 + 48);
  if ( !v66 )
    goto LABEL_111;
  v67 = v66 - 1;
  v8 = *(_QWORD *)(v92 + 32);
  a3 = v86;
  LODWORD(v68) = v67 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v45 = *(_DWORD *)(v92 + 40) + 1;
  v11 = (__int64 *)(v8 + 16LL * (unsigned int)v68);
  v69 = *v11;
  if ( v6 != *v11 )
  {
    v70 = 1;
    v9 = 0;
    while ( v69 != -4096 )
    {
      if ( !v9 && v69 == -8192 )
        v9 = (__int64)v11;
      v68 = v67 & (unsigned int)(v68 + v70);
      v11 = (__int64 *)(v8 + 16 * v68);
      v69 = *v11;
      if ( v6 == *v11 )
        goto LABEL_40;
      ++v70;
    }
    if ( v9 )
      v11 = (__int64 *)v9;
  }
LABEL_40:
  *(_DWORD *)(v3 + 40) = v45;
  if ( *v11 != -4096 )
    --*(_DWORD *)(v3 + 44);
  *v11 = v6;
  v15 = 0;
  v16 = 0;
  v11[1] = 0;
LABEL_4:
  *(_QWORD *)(v15 + 56) = a2;
  v17 = *(unsigned int *)(a3 + 8);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v82 = v3;
    v91 = a3;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v17 + 1, 8u, v8, v9);
    a3 = v91;
    v3 = v82;
    v17 = *(unsigned int *)(v91 + 8);
  }
  v18 = v3;
  v19 = (unsigned int *)a3;
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v16;
  v20 = v96;
  ++*(_DWORD *)(a3 + 8);
  v21 = v95;
  if ( !v20 )
    goto LABEL_27;
  do
  {
    v22 = v20--;
    v23 = 1;
    v24 = v21[v22 - 1];
    LODWORD(v96) = v20;
    v90 = *(_DWORD *)(v24 + 40) & 0xFFFFFF;
    if ( v90 == 1 )
      continue;
    do
    {
      v25 = *(_QWORD *)(v24 + 32);
      v26 = *(_DWORD *)(v18 + 48);
      v27 = *(_DWORD *)(v25 + 40LL * v23 + 8);
      v28 = *(_QWORD *)(v25 + 40LL * (v23 + 1) + 24);
      if ( !v26 )
      {
        ++*(_QWORD *)(v18 + 24);
LABEL_58:
        v77 = v19;
        v84 = v27;
        sub_356EA90(v89, 2 * v26);
        v52 = *(_DWORD *)(v18 + 48);
        if ( !v52 )
          goto LABEL_110;
        v53 = v52 - 1;
        v54 = *(_QWORD *)(v18 + 32);
        v27 = v84;
        v19 = v77;
        v55 = v53 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v48 = *(_DWORD *)(v18 + 40) + 1;
        v46 = (_QWORD *)(v54 + 16LL * v55);
        v56 = *v46;
        if ( v28 != *v46 )
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != -4096 )
          {
            if ( !v58 && v56 == -8192 )
              v58 = v46;
            v55 = v53 & (v57 + v55);
            v46 = (_QWORD *)(v54 + 16LL * v55);
            v56 = *v46;
            if ( v28 == *v46 )
              goto LABEL_49;
            ++v57;
          }
          if ( v58 )
            v46 = v58;
        }
        goto LABEL_49;
      }
      v29 = *(_QWORD *)(v18 + 32);
      v30 = (v26 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v31 = (_QWORD *)(v29 + 16LL * v30);
      v32 = *v31;
      if ( v28 == *v31 )
      {
LABEL_10:
        v33 = v31[1];
        goto LABEL_11;
      }
      v83 = 1;
      v46 = 0;
      while ( v32 != -4096 )
      {
        if ( !v46 && v32 == -8192 )
          v46 = v31;
        v30 = (v26 - 1) & (v83 + v30);
        v31 = (_QWORD *)(v29 + 16LL * v30);
        v32 = *v31;
        if ( v28 == *v31 )
          goto LABEL_10;
        ++v83;
      }
      if ( !v46 )
        v46 = v31;
      v47 = *(_DWORD *)(v18 + 40);
      ++*(_QWORD *)(v18 + 24);
      v48 = v47 + 1;
      if ( 4 * v48 >= 3 * v26 )
        goto LABEL_58;
      if ( v26 - *(_DWORD *)(v18 + 44) - v48 <= v26 >> 3 )
      {
        v78 = v19;
        v85 = v27;
        sub_356EA90(v89, v26);
        v59 = *(_DWORD *)(v18 + 48);
        if ( v59 )
        {
          v60 = v59 - 1;
          v61 = *(_QWORD *)(v18 + 32);
          v62 = 0;
          v63 = v60 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v27 = v85;
          v19 = v78;
          v64 = 1;
          v48 = *(_DWORD *)(v18 + 40) + 1;
          v46 = (_QWORD *)(v61 + 16LL * v63);
          v65 = *v46;
          if ( v28 != *v46 )
          {
            while ( v65 != -4096 )
            {
              if ( v65 == -8192 && !v62 )
                v62 = v46;
              v63 = v60 & (v64 + v63);
              v46 = (_QWORD *)(v61 + 16LL * v63);
              v65 = *v46;
              if ( v28 == *v46 )
                goto LABEL_49;
              ++v64;
            }
            if ( v62 )
              v46 = v62;
          }
          goto LABEL_49;
        }
LABEL_110:
        v3 = v18;
LABEL_111:
        ++*(_DWORD *)(v3 + 40);
        BUG();
      }
LABEL_49:
      *(_DWORD *)(v18 + 40) = v48;
      if ( *v46 != -4096 )
        --*(_DWORD *)(v18 + 44);
      *v46 = v28;
      v33 = 0;
      v46[1] = 0;
LABEL_11:
      v34 = *(_QWORD *)(v33 + 16);
      v35 = *(_DWORD *)(v34 + 8);
      if ( v35 )
      {
        if ( v27 != v35 )
          goto LABEL_13;
      }
      else
      {
        v81 = v19;
        v41 = sub_2EBEE10(*(_QWORD *)(*(_QWORD *)v18 + 40LL), v27);
        v19 = v81;
        if ( !v41 || *(_WORD *)(v41 + 68) && *(_WORD *)(v41 + 68) != 68 || *(_QWORD *)v34 != *(_QWORD *)(v41 + 24) )
        {
LABEL_13:
          v36 = v19;
          if ( v95 != v97 )
          {
            v94 = v19;
            _libc_free((unsigned __int64)v95);
            v36 = v94;
          }
          v37 = *(__int64 **)v36;
          v38 = *(_QWORD *)v36 + 8LL * v36[2];
          if ( *(_QWORD *)v36 != v38 )
          {
            do
            {
              v39 = *v37++;
              *(_QWORD *)(v39 + 56) = 0;
            }
            while ( (__int64 *)v38 != v37 );
          }
          v36[2] = 0;
          return 0;
        }
        v43 = *(_QWORD *)(v34 + 56);
        if ( v43 )
        {
          if ( v43 != v41 )
            goto LABEL_13;
        }
        else
        {
          *(_QWORD *)(v34 + 56) = v41;
          v49 = v81[2];
          if ( v49 + 1 > (unsigned __int64)v81[3] )
          {
            v80 = v41;
            sub_C8D5F0((__int64)v81, v81 + 4, v49 + 1, 8u, v49 + 1, v42);
            v19 = v81;
            v41 = v80;
            v49 = v81[2];
          }
          *(_QWORD *)(*(_QWORD *)v19 + 8 * v49) = v34;
          v50 = (unsigned int)v96;
          v51 = HIDWORD(v96);
          ++v19[2];
          if ( v50 + 1 > v51 )
          {
            v79 = v19;
            v88 = v41;
            sub_C8D5F0((__int64)&v95, v97, v50 + 1, 8u, v50 + 1, v42);
            v50 = (unsigned int)v96;
            v19 = v79;
            v41 = v88;
          }
          v95[v50] = v41;
          LODWORD(v96) = v96 + 1;
        }
      }
      v23 += 2;
    }
    while ( v90 != v23 );
    v20 = v96;
    v21 = v95;
  }
  while ( v20 );
LABEL_27:
  if ( v21 != v97 )
    _libc_free((unsigned __int64)v21);
  return 1;
}
