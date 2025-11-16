// Function: sub_BC1CD0
// Address: 0xbc1cd0
//
__int64 __fastcall sub_BC1CD0(__int64 a1, void *a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rdi
  unsigned int v6; // ebx
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // edx
  unsigned int v10; // eax
  int v11; // ebx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  void **v14; // rdx
  unsigned int i; // r10d
  void **v16; // r8
  void *v17; // r9
  unsigned int v18; // r10d
  _QWORD *v19; // rax
  int v21; // ecx
  int v22; // r8d
  unsigned int v23; // esi
  __int64 v24; // rcx
  unsigned int v25; // edx
  void **v26; // rax
  void *v27; // r8
  void *v28; // r12
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  _QWORD *v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // rdi
  _QWORD *v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rsi
  unsigned int v38; // esi
  __int64 v39; // r8
  __int64 v40; // rdi
  int v41; // r11d
  unsigned int v42; // ecx
  __int64 *v43; // r13
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  _QWORD *v48; // rbx
  _QWORD *v49; // rax
  _QWORD *v50; // r15
  _QWORD *v51; // rax
  __int64 v52; // rdi
  _QWORD *v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // rcx
  int v58; // r9d
  unsigned int m; // eax
  void **v60; // rdx
  unsigned int v61; // eax
  int v62; // ecx
  int v63; // ecx
  int v64; // eax
  int v65; // edi
  __int64 v66; // rsi
  unsigned int v67; // edx
  int v68; // r10d
  __int64 *v69; // r9
  int v70; // edx
  __int64 v71; // rcx
  void **v72; // r9
  int v73; // r8d
  int v74; // edi
  unsigned int j; // eax
  void *v76; // rsi
  unsigned int v77; // eax
  int v78; // edx
  int v79; // ecx
  __int64 v80; // rdi
  int v81; // r8d
  unsigned int k; // eax
  void *v83; // rsi
  unsigned int v84; // eax
  int v85; // eax
  int v86; // r9d
  int v87; // eax
  int v88; // edx
  __int64 v89; // rdi
  unsigned int v90; // ebx
  int v91; // r9d
  __int64 v92; // rsi
  __int64 v93; // [rsp+8h] [rbp-58h]
  __int64 v94; // [rsp+8h] [rbp-58h]
  __int64 v95; // [rsp+10h] [rbp-50h]
  __int64 *v96; // [rsp+10h] [rbp-50h]
  int v97; // [rsp+10h] [rbp-50h]
  _QWORD v99[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a1;
  v4 = a1 + 64;
  v6 = (unsigned int)a2;
  v7 = *(_DWORD *)(v3 + 88);
  if ( !v7 )
  {
    ++*(_QWORD *)(v3 + 64);
    goto LABEL_90;
  }
  v8 = *(_QWORD *)(v3 + 72);
  v9 = v6 >> 9;
  v10 = v6 >> 4;
  v11 = 1;
  v12 = 0xBF58476D1CE4E5B9LL
      * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(v9 ^ v10) << 32));
  v13 = (v12 >> 31) ^ v12;
  v14 = 0;
  for ( i = v13 & (v7 - 1); ; i = (v7 - 1) & v18 )
  {
    v16 = (void **)(v8 + 24LL * i);
    v17 = *v16;
    if ( *v16 == a2 && (void *)a3 == v16[1] )
    {
      v19 = v16[2];
      return v19[3];
    }
    if ( v17 == (void *)-4096LL )
      break;
    if ( v17 == (void *)-8192LL && v16[1] == (void *)-8192LL && !v14 )
      v14 = (void **)(v8 + 24LL * i);
LABEL_9:
    v18 = v11 + i;
    ++v11;
  }
  if ( v16[1] != (void *)-4096LL )
    goto LABEL_9;
  v21 = *(_DWORD *)(v3 + 80);
  if ( !v14 )
    v14 = v16;
  ++*(_QWORD *)(v3 + 64);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v7 )
  {
LABEL_90:
    sub_BC1A00(v4, 2 * v7);
    v70 = *(_DWORD *)(v3 + 88);
    if ( v70 )
    {
      v72 = 0;
      v73 = 1;
      v74 = v70 - 1;
      for ( j = (v70 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v74 & v77 )
      {
        v71 = *(_QWORD *)(v3 + 72);
        v14 = (void **)(v71 + 24LL * j);
        v76 = *v14;
        if ( *v14 == a2 && (void *)a3 == v14[1] )
          break;
        if ( v76 == (void *)-4096LL )
        {
          if ( v14[1] == (void *)-4096LL )
          {
LABEL_123:
            if ( v72 )
              v14 = v72;
            v22 = *(_DWORD *)(v3 + 80) + 1;
            goto LABEL_18;
          }
        }
        else if ( v76 == (void *)-8192LL && v14[1] == (void *)-8192LL && !v72 )
        {
          v72 = (void **)(v71 + 24LL * j);
        }
        v77 = v73 + j;
        ++v73;
      }
      goto LABEL_119;
    }
LABEL_138:
    ++*(_DWORD *)(v3 + 80);
    BUG();
  }
  if ( v7 - *(_DWORD *)(v3 + 84) - v22 <= v7 >> 3 )
  {
    v97 = v13;
    sub_BC1A00(v4, v7);
    v78 = *(_DWORD *)(v3 + 88);
    if ( v78 )
    {
      v79 = v78 - 1;
      v72 = 0;
      v81 = 1;
      for ( k = (v78 - 1) & v97; ; k = v79 & v84 )
      {
        v80 = *(_QWORD *)(v3 + 72);
        v14 = (void **)(v80 + 24LL * k);
        v83 = *v14;
        if ( *v14 == a2 && (void *)a3 == v14[1] )
          break;
        if ( v83 == (void *)-4096LL )
        {
          if ( v14[1] == (void *)-4096LL )
            goto LABEL_123;
        }
        else if ( v83 == (void *)-8192LL && v14[1] == (void *)-8192LL && !v72 )
        {
          v72 = (void **)(v80 + 24LL * k);
        }
        v84 = v81 + k;
        ++v81;
      }
LABEL_119:
      v22 = *(_DWORD *)(v3 + 80) + 1;
      goto LABEL_18;
    }
    goto LABEL_138;
  }
LABEL_18:
  *(_DWORD *)(v3 + 80) = v22;
  if ( *v14 != (void *)-4096LL || v14[1] != (void *)-4096LL )
    --*(_DWORD *)(v3 + 84);
  v14[1] = (void *)a3;
  v14[2] = 0;
  *v14 = a2;
  v23 = *(_DWORD *)(v3 + 24);
  v24 = *(_QWORD *)(v3 + 8);
  if ( !v23 )
  {
LABEL_84:
    v26 = (void **)(v24 + 16LL * v23);
    goto LABEL_22;
  }
  v25 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (void **)(v24 + 16LL * v25);
  v27 = *v26;
  if ( *v26 != a2 )
  {
    v85 = 1;
    while ( v27 != (void *)-4096LL )
    {
      v86 = v85 + 1;
      v25 = (v23 - 1) & (v85 + v25);
      v26 = (void **)(v24 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == a2 )
        goto LABEL_22;
      v85 = v86;
    }
    goto LABEL_84;
  }
LABEL_22:
  v28 = v26[1];
  if ( a2 == &unk_4F8A320 || (v29 = *(_QWORD *)(sub_BC1CD0(v3, &unk_4F8A320) + 8), (v93 = v29) == 0) )
  {
    v38 = *(_DWORD *)(v3 + 56);
    v93 = 0;
    v39 = v3 + 32;
    if ( v38 )
      goto LABEL_37;
    goto LABEL_73;
  }
  v30 = *(_QWORD **)(v29 + 720);
  v31 = &v30[4 * *(unsigned int *)(v29 + 728)];
  if ( v30 != v31 )
  {
    v95 = v3;
    v32 = v31;
    do
    {
      v99[0] = 0;
      v33 = (_QWORD *)sub_22077B0(16);
      if ( v33 )
      {
        v33[1] = a3;
        *v33 = &unk_49DB0A8;
      }
      v34 = v99[0];
      v99[0] = v33;
      if ( v34 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
      v35 = v30;
      v37 = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v28 + 24LL))(v28);
      if ( (v30[3] & 2) == 0 )
        v35 = (_QWORD *)*v30;
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v30[3] & 0xFFFFFFFFFFFFFFF8LL))(v35, v37, v36, v99);
      if ( v99[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v99[0] + 8LL))(v99[0]);
      v30 += 4;
    }
    while ( v32 != v30 );
    v3 = v95;
  }
  v38 = *(_DWORD *)(v3 + 56);
  v39 = v3 + 32;
  if ( !v38 )
  {
LABEL_73:
    ++*(_QWORD *)(v3 + 32);
    goto LABEL_74;
  }
LABEL_37:
  v40 = *(_QWORD *)(v3 + 40);
  v41 = 1;
  v42 = (v38 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v43 = (__int64 *)(v40 + 32LL * v42);
  v44 = 0;
  v45 = *v43;
  if ( a3 == *v43 )
  {
LABEL_38:
    v96 = v43 + 1;
    goto LABEL_39;
  }
  while ( v45 != -4096 )
  {
    if ( v45 == -8192 && !v44 )
      v44 = v43;
    v42 = (v38 - 1) & (v41 + v42);
    v43 = (__int64 *)(v40 + 32LL * v42);
    v45 = *v43;
    if ( a3 == *v43 )
      goto LABEL_38;
    ++v41;
  }
  v62 = *(_DWORD *)(v3 + 48);
  if ( !v44 )
    v44 = v43;
  ++*(_QWORD *)(v3 + 32);
  v63 = v62 + 1;
  if ( 4 * v63 >= 3 * v38 )
  {
LABEL_74:
    sub_BBC880(v39, 2 * v38);
    v64 = *(_DWORD *)(v3 + 56);
    if ( v64 )
    {
      v65 = v64 - 1;
      v66 = *(_QWORD *)(v3 + 40);
      v67 = (v64 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v63 = *(_DWORD *)(v3 + 48) + 1;
      v44 = (__int64 *)(v66 + 32LL * v67);
      v39 = *v44;
      if ( a3 != *v44 )
      {
        v68 = 1;
        v69 = 0;
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v69 )
            v69 = v44;
          v67 = v65 & (v68 + v67);
          v44 = (__int64 *)(v66 + 32LL * v67);
          v39 = *v44;
          if ( a3 == *v44 )
            goto LABEL_69;
          ++v68;
        }
        if ( v69 )
          v44 = v69;
      }
      goto LABEL_69;
    }
    goto LABEL_139;
  }
  if ( v38 - *(_DWORD *)(v3 + 52) - v63 <= v38 >> 3 )
  {
    sub_BBC880(v39, v38);
    v87 = *(_DWORD *)(v3 + 56);
    if ( v87 )
    {
      v88 = v87 - 1;
      v89 = *(_QWORD *)(v3 + 40);
      v39 = 0;
      v90 = (v87 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v91 = 1;
      v63 = *(_DWORD *)(v3 + 48) + 1;
      v44 = (__int64 *)(v89 + 32LL * v90);
      v92 = *v44;
      if ( a3 != *v44 )
      {
        while ( v92 != -4096 )
        {
          if ( !v39 && v92 == -8192 )
            v39 = (__int64)v44;
          v90 = v88 & (v91 + v90);
          v44 = (__int64 *)(v89 + 32LL * v90);
          v92 = *v44;
          if ( a3 == *v44 )
            goto LABEL_69;
          ++v91;
        }
        if ( v39 )
          v44 = (__int64 *)v39;
      }
      goto LABEL_69;
    }
LABEL_139:
    ++*(_DWORD *)(v3 + 48);
    BUG();
  }
LABEL_69:
  *(_DWORD *)(v3 + 48) = v63;
  if ( *v44 != -4096 )
    --*(_DWORD *)(v3 + 52);
  *v44 = a3;
  v96 = v44 + 1;
  v44[2] = (__int64)(v44 + 1);
  v44[1] = (__int64)(v44 + 1);
  v44[3] = 0;
LABEL_39:
  (*(void (__fastcall **)(_QWORD *, void *, __int64, __int64, __int64))(*(_QWORD *)v28 + 16LL))(v99, v28, a3, v3, v39);
  v46 = sub_22077B0(32);
  *(_QWORD *)(v46 + 16) = a2;
  v47 = v99[0];
  v99[0] = 0;
  *(_QWORD *)(v46 + 24) = v47;
  sub_2208C80(v46, v96);
  ++v96[2];
  if ( v99[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v99[0] + 8LL))(v99[0]);
  if ( v93 )
  {
    v48 = *(_QWORD **)(v93 + 864);
    v49 = &v48[4 * *(unsigned int *)(v93 + 872)];
    if ( v48 != v49 )
    {
      v94 = v3;
      v50 = v49;
      do
      {
        v99[0] = 0;
        v51 = (_QWORD *)sub_22077B0(16);
        if ( v51 )
        {
          v51[1] = a3;
          *v51 = &unk_49DB0A8;
        }
        v52 = v99[0];
        v99[0] = v51;
        if ( v52 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v52 + 8LL))(v52);
        v53 = v48;
        v55 = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v28 + 24LL))(v28);
        if ( (v48[3] & 2) == 0 )
          v53 = (_QWORD *)*v48;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v48[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v53,
          v55,
          v54,
          v99);
        if ( v99[0] )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v99[0] + 8LL))(v99[0]);
        v48 += 4;
      }
      while ( v50 != v48 );
      v3 = v94;
    }
  }
  v56 = *(unsigned int *)(v3 + 88);
  v57 = *(_QWORD *)(v3 + 72);
  if ( (_DWORD)v56 )
  {
    v58 = 1;
    for ( m = (v56 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = (v56 - 1) & v61 )
    {
      v60 = (void **)(v57 + 24LL * m);
      if ( *v60 == a2 && (void *)a3 == v60[1] )
        break;
      if ( *v60 == (void *)-4096LL && v60[1] == (void *)-4096LL )
        goto LABEL_82;
      v61 = v58 + m;
      ++v58;
    }
  }
  else
  {
LABEL_82:
    v60 = (void **)(v57 + 24 * v56);
  }
  v19 = (_QWORD *)v96[1];
  v60[2] = v19;
  return v19[3];
}
