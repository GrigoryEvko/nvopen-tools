// Function: sub_31E2440
// Address: 0x31e2440
//
void __fastcall sub_31E2440(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // r12
  bool v7; // r14
  unsigned int v8; // esi
  __int64 v9; // r9
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // edi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edi
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  unsigned int v20; // eax
  int v21; // eax
  int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // eax
  _QWORD *v25; // r14
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  bool v31; // r14
  unsigned int v32; // esi
  __int64 v33; // r8
  int v34; // eax
  __int64 v35; // rdx
  int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdi
  unsigned int v40; // ecx
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rax
  bool v47; // zf
  __int64 v48; // rdx
  bool v49; // al
  __int64 v50; // rax
  __int64 v51; // rbx
  __int64 v52; // rax
  unsigned __int64 v53; // rbx
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // r13
  __int64 v57; // rcx
  unsigned __int64 *v58; // rax
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rcx
  int v61; // edi
  int v62; // ecx
  __int64 v63; // rdi
  int v64; // r10d
  unsigned int v65; // eax
  __int64 v66; // rsi
  int v67; // eax
  __int64 v68; // rdi
  int v69; // r10d
  unsigned int v70; // esi
  __int64 v71; // r8
  int v72; // r11d
  int v73; // eax
  int v74; // eax
  int v75; // eax
  __int64 v76; // rdi
  int v77; // r10d
  unsigned int v78; // esi
  __int64 v79; // r8
  int v80; // r11d
  int v81; // eax
  int v82; // eax
  int v83; // eax
  __int64 v84; // rdi
  int v85; // r10d
  unsigned int v86; // ecx
  __int64 v87; // rsi
  int v88; // edx
  __int64 v89; // rax
  __int64 v90; // [rsp+0h] [rbp-B0h]
  __int64 v91; // [rsp+0h] [rbp-B0h]
  __int64 v92; // [rsp+8h] [rbp-A8h]
  __int64 v93; // [rsp+8h] [rbp-A8h]
  __int64 v94; // [rsp+8h] [rbp-A8h]
  __int64 v95; // [rsp+8h] [rbp-A8h]
  __int64 v96; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+18h] [rbp-98h]
  unsigned int v98; // [rsp+20h] [rbp-90h]
  _QWORD v99[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v100; // [rsp+40h] [rbp-70h]
  __int64 (__fastcall **v101)(); // [rsp+50h] [rbp-60h] BYREF
  __int64 v102; // [rsp+58h] [rbp-58h] BYREF
  __int64 v103; // [rsp+60h] [rbp-50h]
  __int64 v104; // [rsp+68h] [rbp-48h]
  __int64 v105; // [rsp+70h] [rbp-40h]

  v3 = a1 + 8;
  v5 = a2;
  v103 = a2;
  v101 = 0;
  v102 = 0;
  v7 = a2 != -8192 && a2 != 0 && a2 != -4096;
  if ( v7 )
    sub_BD73F0((__int64)&v101);
  v8 = *(_DWORD *)(a1 + 32);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_5:
    sub_31E01F0(v3, 2 * v8);
    v10 = *(_DWORD *)(a1 + 32);
    if ( !v10 )
    {
LABEL_6:
      v11 = v103;
      v12 = 0;
LABEL_7:
      v13 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_8;
    }
    v11 = v103;
    v67 = v10 - 1;
    v68 = *(_QWORD *)(a1 + 16);
    v9 = 0;
    v69 = 1;
    v70 = v67 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
    v12 = v68 + 48LL * v70;
    v71 = *(_QWORD *)(v12 + 16);
    if ( v103 == v71 )
      goto LABEL_7;
    while ( v71 != -4096 )
    {
      if ( v71 == -8192 && !v9 )
        v9 = v12;
      v70 = v67 & (v69 + v70);
      v12 = v68 + 48LL * v70;
      v71 = *(_QWORD *)(v12 + 16);
      if ( v103 == v71 )
        goto LABEL_7;
      ++v69;
    }
LABEL_122:
    if ( v9 )
      v12 = v9;
    goto LABEL_7;
  }
  v11 = v103;
  v9 = *(_QWORD *)(a1 + 16);
  v16 = (v8 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
  v17 = v9 + 48LL * v16;
  v18 = *(_QWORD *)(v17 + 16);
  if ( v103 == v18 )
  {
LABEL_18:
    v15 = v17 + 24;
    goto LABEL_19;
  }
  v72 = 1;
  v12 = 0;
  while ( v18 != -4096 )
  {
    if ( !v12 && v18 == -8192 )
      v12 = v17;
    v16 = (v8 - 1) & (v72 + v16);
    v17 = v9 + 48LL * v16;
    v18 = *(_QWORD *)(v17 + 16);
    if ( v103 == v18 )
      goto LABEL_18;
    ++v72;
  }
  if ( !v12 )
    v12 = v17;
  v73 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v13 = v73 + 1;
  if ( 4 * (v73 + 1) >= 3 * v8 )
    goto LABEL_5;
  if ( v8 - *(_DWORD *)(a1 + 28) - v13 <= v8 >> 3 )
  {
    sub_31E01F0(v3, v8);
    v74 = *(_DWORD *)(a1 + 32);
    if ( !v74 )
      goto LABEL_6;
    v11 = v103;
    v75 = v74 - 1;
    v76 = *(_QWORD *)(a1 + 16);
    v9 = 0;
    v77 = 1;
    v78 = v75 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
    v12 = v76 + 48LL * v78;
    v79 = *(_QWORD *)(v12 + 16);
    if ( v103 == v79 )
      goto LABEL_7;
    while ( v79 != -4096 )
    {
      if ( !v9 && v79 == -8192 )
        v9 = v12;
      v78 = v75 & (v77 + v78);
      v12 = v76 + 48LL * v78;
      v79 = *(_QWORD *)(v12 + 16);
      if ( v103 == v79 )
        goto LABEL_7;
      ++v77;
    }
    goto LABEL_122;
  }
LABEL_8:
  *(_DWORD *)(a1 + 24) = v13;
  if ( *(_QWORD *)(v12 + 16) == -4096 )
  {
    if ( v11 != -4096 )
    {
LABEL_13:
      *(_QWORD *)(v12 + 16) = v11;
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
      {
        v93 = v12;
        sub_BD73F0(v12);
        v12 = v93;
      }
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 28);
    v14 = *(_QWORD *)(v12 + 16);
    if ( v11 != v14 )
    {
      if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
      {
        v90 = v11;
        v92 = v12;
        sub_BD60C0((_QWORD *)v12);
        v11 = v90;
        v12 = v92;
      }
      goto LABEL_13;
    }
  }
  *(_QWORD *)(v12 + 40) = 0;
  v15 = v12 + 24;
  *(_OWORD *)(v12 + 24) = 0;
LABEL_19:
  v96 = *(_QWORD *)v15;
  v19 = *(_QWORD *)(v15 + 8);
  *(_QWORD *)v15 = 0;
  v20 = *(_DWORD *)(v15 + 16);
  v97 = v19;
  v98 = v20;
  if ( v103 != 0 && v103 != -4096 && v103 != -8192 )
    sub_BD60C0(&v101);
  v99[0] = 0;
  v99[1] = 0;
  v100 = v5;
  if ( v7 )
  {
    sub_BD73F0((__int64)v99);
    v5 = v100;
  }
  v21 = *(_DWORD *)(a1 + 32);
  if ( v21 )
  {
    v22 = v21 - 1;
    v23 = *(_QWORD *)(a1 + 16);
    v24 = (v21 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v25 = (_QWORD *)(v23 + 48LL * v24);
    v26 = v25[2];
    if ( v26 == v5 )
    {
LABEL_26:
      v27 = v25[3];
      if ( v27 )
      {
        if ( (v27 & 4) != 0 )
        {
          v28 = (unsigned __int64 *)(v27 & 0xFFFFFFFFFFFFFFF8LL);
          v29 = (unsigned __int64)v28;
          if ( v28 )
          {
            if ( (unsigned __int64 *)*v28 != v28 + 2 )
              _libc_free(*v28);
            j_j___libc_free_0(v29);
          }
        }
      }
      v101 = 0;
      v102 = 0;
      v103 = -8192;
      v30 = v25[2];
      if ( v30 != -8192 )
      {
        if ( v30 != -4096 && v30 )
          sub_BD60C0(v25);
        v25[2] = -8192;
        if ( v103 != 0 && v103 != -4096 && v103 != -8192 )
          sub_BD60C0(&v101);
      }
      --*(_DWORD *)(a1 + 24);
      v5 = v100;
      ++*(_DWORD *)(a1 + 28);
    }
    else
    {
      v61 = 1;
      while ( v26 != -4096 )
      {
        v24 = v22 & (v61 + v24);
        v25 = (_QWORD *)(v23 + 48LL * v24);
        v26 = v25[2];
        if ( v26 == v5 )
          goto LABEL_26;
        ++v61;
      }
    }
  }
  if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
    sub_BD60C0(v99);
  v101 = 0;
  v102 = 0;
  v103 = a3;
  v31 = a3 != 0 && a3 != -4096 && a3 != -8192;
  if ( v31 )
    sub_BD73F0((__int64)&v101);
  v32 = *(_DWORD *)(a1 + 32);
  if ( !v32 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_47:
    sub_31E01F0(v3, 2 * v32);
    v34 = *(_DWORD *)(a1 + 32);
    if ( !v34 )
    {
LABEL_48:
      v35 = v103;
      v9 = 0;
LABEL_49:
      v36 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_50;
    }
    v35 = v103;
    v62 = v34 - 1;
    v63 = *(_QWORD *)(a1 + 16);
    v33 = 0;
    v64 = 1;
    v65 = (v34 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
    v9 = v63 + 48LL * v65;
    v66 = *(_QWORD *)(v9 + 16);
    if ( v66 == v103 )
      goto LABEL_49;
    while ( v66 != -4096 )
    {
      if ( v66 == -8192 && !v33 )
        v33 = v9;
      v65 = v62 & (v64 + v65);
      v9 = v63 + 48LL * v65;
      v66 = *(_QWORD *)(v9 + 16);
      if ( v103 == v66 )
        goto LABEL_49;
      ++v64;
    }
LABEL_117:
    if ( v33 )
      v9 = v33;
    goto LABEL_49;
  }
  v35 = v103;
  v33 = v32 - 1;
  v39 = *(_QWORD *)(a1 + 16);
  v40 = v33 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
  v41 = v39 + 48LL * v40;
  v42 = *(_QWORD *)(v41 + 16);
  if ( v103 == v42 )
  {
LABEL_60:
    v38 = v41 + 24;
    goto LABEL_61;
  }
  v80 = 1;
  v9 = 0;
  while ( v42 != -4096 )
  {
    if ( !v9 && v42 == -8192 )
      v9 = v41;
    v40 = v33 & (v80 + v40);
    v41 = v39 + 48LL * v40;
    v42 = *(_QWORD *)(v41 + 16);
    if ( v103 == v42 )
      goto LABEL_60;
    ++v80;
  }
  v81 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
    v9 = v41;
  ++*(_QWORD *)(a1 + 8);
  v36 = v81 + 1;
  if ( 4 * (v81 + 1) >= 3 * v32 )
    goto LABEL_47;
  if ( v32 - *(_DWORD *)(a1 + 28) - v36 <= v32 >> 3 )
  {
    sub_31E01F0(v3, v32);
    v82 = *(_DWORD *)(a1 + 32);
    if ( !v82 )
      goto LABEL_48;
    v35 = v103;
    v83 = v82 - 1;
    v84 = *(_QWORD *)(a1 + 16);
    v33 = 0;
    v85 = 1;
    v86 = v83 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
    v9 = v84 + 48LL * v86;
    v87 = *(_QWORD *)(v9 + 16);
    if ( v103 == v87 )
      goto LABEL_49;
    while ( v87 != -4096 )
    {
      if ( !v33 && v87 == -8192 )
        v33 = v9;
      v86 = v83 & (v85 + v86);
      v9 = v84 + 48LL * v86;
      v87 = *(_QWORD *)(v9 + 16);
      if ( v103 == v87 )
        goto LABEL_49;
      ++v85;
    }
    goto LABEL_117;
  }
LABEL_50:
  *(_DWORD *)(a1 + 24) = v36;
  if ( *(_QWORD *)(v9 + 16) == -4096 )
  {
    if ( v35 != -4096 )
    {
LABEL_55:
      *(_QWORD *)(v9 + 16) = v35;
      if ( v35 == 0 || v35 == -4096 || v35 == -8192 )
      {
        v35 = v103;
      }
      else
      {
        v95 = v9;
        sub_BD73F0(v9);
        v35 = v103;
        v9 = v95;
      }
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 28);
    v37 = *(_QWORD *)(v9 + 16);
    if ( v35 != v37 )
    {
      if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
      {
        v91 = v35;
        v94 = v9;
        sub_BD60C0((_QWORD *)v9);
        v35 = v91;
        v9 = v94;
      }
      goto LABEL_55;
    }
  }
  *(_QWORD *)(v9 + 40) = 0;
  v38 = v9 + 24;
  *(_OWORD *)(v9 + 24) = 0;
LABEL_61:
  if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
    sub_BD60C0(&v101);
  if ( (*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL) != 0
    && ((*(_QWORD *)v38 & 4) == 0 || *(_DWORD *)((*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL) + 8)) )
  {
    v102 = 2;
    v101 = off_4A35038;
    v103 = 0;
    v43 = *(_QWORD *)(a1 + 40);
    v104 = 0;
    v105 = 0;
    v44 = (_QWORD *)(v43 + 40LL * v98);
    v45 = v44[3];
    if ( !v45 )
    {
      v44[4] = 0;
      goto LABEL_75;
    }
    if ( v45 == -4096 || v45 == -8192 )
    {
      v44[3] = 0;
      v49 = v104 != -8192 && v104 != -4096 && v104 != 0;
      v48 = 0;
    }
    else
    {
      sub_BD60C0(v44 + 1);
      v46 = v104;
      v47 = v104 == 0;
      v44[3] = v104;
      if ( v46 == -4096 || v47 || v46 == -8192 )
      {
        v44[4] = v105;
        goto LABEL_75;
      }
      sub_BD6050(v44 + 1, v102 & 0xFFFFFFFFFFFFFFF8LL);
      v48 = v105;
      v49 = v104 != -8192 && v104 != 0 && v104 != -4096;
    }
    v44[4] = v48;
    v101 = (__int64 (__fastcall **)())&unk_49DB368;
    if ( v49 )
      sub_BD60C0(&v102);
LABEL_75:
    sub_31DF770((__int64 *)v38, &v96);
    v50 = v96;
    if ( !v96 )
      return;
LABEL_98:
    if ( (v50 & 4) != 0 )
    {
      v58 = (unsigned __int64 *)(v50 & 0xFFFFFFFFFFFFFFF8LL);
      v59 = (unsigned __int64)v58;
      if ( v58 )
      {
        if ( (unsigned __int64 *)*v58 != v58 + 2 )
          _libc_free(*v58);
        j_j___libc_free_0(v59);
      }
    }
    return;
  }
  v51 = *(_QWORD *)(a1 + 40) + 40LL * v98;
  v52 = *(_QWORD *)(v51 + 24);
  if ( a3 != v52 )
  {
    if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
      sub_BD60C0((_QWORD *)(v51 + 8));
    *(_QWORD *)(v51 + 24) = a3;
    if ( v31 )
      sub_BD73F0(v51 + 8);
  }
  v50 = v96;
  if ( (__int64 *)v38 == &v96 )
  {
LABEL_97:
    *(_QWORD *)(v38 + 8) = v97;
    *(_DWORD *)(v38 + 16) = v98;
    if ( !v50 )
      return;
    goto LABEL_98;
  }
  v53 = v96 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v96 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v96 & 4) != 0 && !*(_DWORD *)(v53 + 8) )
  {
    v57 = *(_QWORD *)v38;
    if ( ((*(__int64 *)v38 >> 2) & 1) != 0 )
    {
      if ( v57 )
      {
        if ( ((*(__int64 *)v38 >> 2) & 1) != 0 )
        {
          v60 = v57 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v60 )
            *(_DWORD *)(v60 + 8) = 0;
        }
      }
    }
    else
    {
      *(_QWORD *)v38 = 0;
      v50 = v96;
    }
    goto LABEL_97;
  }
  v54 = *(_QWORD *)v38;
  if ( !*(_QWORD *)v38 || (v54 & 4) == 0 || (v55 = v54 & 0xFFFFFFFFFFFFFFF8LL, (v56 = v55) == 0) )
  {
LABEL_94:
    *(_QWORD *)v38 = v96;
    *(_QWORD *)(v38 + 8) = v97;
    *(_DWORD *)(v38 + 16) = v98;
    return;
  }
  if ( ((v96 >> 2) & 1) != 0 )
  {
    if ( *(_QWORD *)v55 != v55 + 16 )
      _libc_free(*(_QWORD *)v55);
    j_j___libc_free_0(v56);
    goto LABEL_94;
  }
  *(_DWORD *)(v55 + 8) = 0;
  v88 = *(_DWORD *)(v55 + 12);
  v89 = 0;
  if ( !v88 )
  {
    sub_C8D5F0(v56, (const void *)(v56 + 16), 1u, 8u, v33, v9);
    v89 = 8LL * *(unsigned int *)(v56 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v56 + v89) = v53;
  ++*(_DWORD *)(v56 + 8);
  *(_QWORD *)(v38 + 8) = v97;
  *(_DWORD *)(v38 + 16) = v98;
}
