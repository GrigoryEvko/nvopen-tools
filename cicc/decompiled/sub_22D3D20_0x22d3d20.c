// Function: sub_22D3D20
// Address: 0x22d3d20
//
__int64 __fastcall sub_22D3D20(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r15
  __int64 v5; // rdi
  unsigned int v7; // ebx
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r11d
  unsigned int v11; // r8d
  unsigned __int64 v12; // rax
  __int64 **v13; // rdx
  unsigned int i; // r9d
  __int64 **v15; // r8
  __int64 *v16; // rbx
  unsigned int v17; // r9d
  __int64 *v18; // rax
  int v20; // ebx
  int v21; // r8d
  unsigned int v22; // esi
  __int64 v23; // rcx
  unsigned int v24; // edx
  __int64 **v25; // rax
  __int64 *v26; // r8
  __int64 *v27; // r12
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // rax
  _QWORD *v31; // r15
  _QWORD *v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rsi
  unsigned int v37; // esi
  __int64 v38; // r8
  __int64 v39; // rdi
  int v40; // r11d
  unsigned int v41; // ecx
  __int64 **v42; // r13
  __int64 **v43; // rax
  __int64 *v44; // rdx
  _QWORD *v45; // rdi
  __int64 v46; // rax
  _QWORD *v47; // rbx
  _QWORD *v48; // r15
  _QWORD *v49; // rax
  __int64 v50; // rdi
  _QWORD *v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rsi
  __int64 v55; // rcx
  int v56; // r9d
  unsigned int m; // eax
  __int64 **v58; // rdx
  unsigned int v59; // eax
  int v60; // ecx
  int v61; // ecx
  int v62; // eax
  int v63; // edi
  __int64 v64; // rsi
  unsigned int v65; // edx
  __int64 *v66; // r8
  int v67; // r10d
  __int64 **v68; // r9
  int v69; // edx
  __int64 v70; // rcx
  __int64 **v71; // r9
  int v72; // r8d
  int v73; // edi
  unsigned int j; // eax
  __int64 *v75; // rsi
  unsigned int v76; // eax
  int v77; // edx
  int v78; // ecx
  __int64 v79; // rdi
  int v80; // r8d
  unsigned int k; // eax
  __int64 *v82; // rsi
  unsigned int v83; // eax
  int v84; // eax
  int v85; // r9d
  int v86; // eax
  int v87; // edx
  __int64 v88; // rdi
  __int64 **v89; // r8
  unsigned int v90; // ebx
  int v91; // r9d
  __int64 *v92; // rsi
  __int64 v93; // [rsp+0h] [rbp-60h]
  __int64 v94; // [rsp+8h] [rbp-58h]
  __int64 v95; // [rsp+8h] [rbp-58h]
  int v96; // [rsp+8h] [rbp-58h]
  __int64 v98; // [rsp+10h] [rbp-50h]
  _QWORD v100[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a1;
  v5 = a1 + 64;
  v7 = (unsigned int)a2;
  v8 = *(_DWORD *)(v4 + 88);
  if ( !v8 )
  {
    ++*(_QWORD *)(v4 + 64);
    goto LABEL_90;
  }
  v9 = *(_QWORD *)(v4 + 72);
  v10 = 1;
  v11 = (unsigned int)a3 >> 9;
  v12 = ((0xBF58476D1CE4E5B9LL * (v11 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)((v7 >> 9) ^ (v7 >> 4)) << 32))) >> 31)
      ^ (0xBF58476D1CE4E5B9LL * (v11 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)((v7 >> 9) ^ (v7 >> 4)) << 32)));
  v13 = 0;
  for ( i = (((0xBF58476D1CE4E5B9LL * (v11 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)((v7 >> 9) ^ (v7 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v11 ^ ((unsigned int)a3 >> 4))))
          & (v8 - 1); ; i = (v8 - 1) & v17 )
  {
    v15 = (__int64 **)(v9 + 24LL * i);
    v16 = *v15;
    if ( *v15 == a2 && a3 == v15[1] )
    {
      v18 = v15[2];
      return v18[3];
    }
    if ( v16 == (__int64 *)-4096LL )
      break;
    if ( v16 == (__int64 *)-8192LL && v15[1] == (__int64 *)-8192LL && !v13 )
      v13 = (__int64 **)(v9 + 24LL * i);
LABEL_9:
    v17 = v10 + i;
    ++v10;
  }
  if ( v15[1] != (__int64 *)-4096LL )
    goto LABEL_9;
  v20 = *(_DWORD *)(v4 + 80);
  if ( !v13 )
    v13 = (__int64 **)(v9 + 24LL * i);
  ++*(_QWORD *)(v4 + 64);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v8 )
  {
LABEL_90:
    sub_22D3A50(v5, 2 * v8);
    v69 = *(_DWORD *)(v4 + 88);
    if ( v69 )
    {
      v71 = 0;
      v72 = 1;
      v73 = v69 - 1;
      for ( j = (v69 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v73 & v76 )
      {
        v70 = *(_QWORD *)(v4 + 72);
        v13 = (__int64 **)(v70 + 24LL * j);
        v75 = *v13;
        if ( *v13 == a2 && a3 == v13[1] )
          break;
        if ( v75 == (__int64 *)-4096LL )
        {
          if ( v13[1] == (__int64 *)-4096LL )
          {
LABEL_123:
            if ( v71 )
              v13 = v71;
            v21 = *(_DWORD *)(v4 + 80) + 1;
            goto LABEL_18;
          }
        }
        else if ( v75 == (__int64 *)-8192LL && v13[1] == (__int64 *)-8192LL && !v71 )
        {
          v71 = (__int64 **)(v70 + 24LL * j);
        }
        v76 = v72 + j;
        ++v72;
      }
      goto LABEL_119;
    }
LABEL_138:
    ++*(_DWORD *)(v4 + 80);
    BUG();
  }
  if ( v8 - *(_DWORD *)(v4 + 84) - v21 <= v8 >> 3 )
  {
    v96 = v12;
    sub_22D3A50(v5, v8);
    v77 = *(_DWORD *)(v4 + 88);
    if ( v77 )
    {
      v78 = v77 - 1;
      v71 = 0;
      v80 = 1;
      for ( k = (v77 - 1) & v96; ; k = v78 & v83 )
      {
        v79 = *(_QWORD *)(v4 + 72);
        v13 = (__int64 **)(v79 + 24LL * k);
        v82 = *v13;
        if ( *v13 == a2 && a3 == v13[1] )
          break;
        if ( v82 == (__int64 *)-4096LL )
        {
          if ( v13[1] == (__int64 *)-4096LL )
            goto LABEL_123;
        }
        else if ( v82 == (__int64 *)-8192LL && v13[1] == (__int64 *)-8192LL && !v71 )
        {
          v71 = (__int64 **)(v79 + 24LL * k);
        }
        v83 = v80 + k;
        ++v80;
      }
LABEL_119:
      v21 = *(_DWORD *)(v4 + 80) + 1;
      goto LABEL_18;
    }
    goto LABEL_138;
  }
LABEL_18:
  *(_DWORD *)(v4 + 80) = v21;
  if ( *v13 != (__int64 *)-4096LL || v13[1] != (__int64 *)-4096LL )
    --*(_DWORD *)(v4 + 84);
  v13[1] = a3;
  v13[2] = 0;
  *v13 = a2;
  v22 = *(_DWORD *)(v4 + 24);
  v23 = *(_QWORD *)(v4 + 8);
  if ( !v22 )
  {
LABEL_84:
    v25 = (__int64 **)(v23 + 16LL * v22);
    goto LABEL_22;
  }
  v24 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (__int64 **)(v23 + 16LL * v24);
  v26 = *v25;
  if ( *v25 != a2 )
  {
    v84 = 1;
    while ( v26 != (__int64 *)-4096LL )
    {
      v85 = v84 + 1;
      v24 = (v22 - 1) & (v84 + v24);
      v25 = (__int64 **)(v23 + 16LL * v24);
      v26 = *v25;
      if ( *v25 == a2 )
        goto LABEL_22;
      v84 = v85;
    }
    goto LABEL_84;
  }
LABEL_22:
  v27 = v25[1];
  if ( a2 == &qword_4F8A320 || (v28 = *(_QWORD *)(sub_22D3D20(v4, &qword_4F8A320, a3, a4) + 8), (v93 = v28) == 0) )
  {
    v37 = *(_DWORD *)(v4 + 56);
    v93 = 0;
    v38 = v4 + 32;
    if ( v37 )
      goto LABEL_37;
    goto LABEL_73;
  }
  v29 = *(_QWORD **)(v28 + 720);
  v30 = &v29[4 * *(unsigned int *)(v28 + 728)];
  if ( v29 != v30 )
  {
    v94 = v4;
    v31 = v30;
    do
    {
      v100[0] = 0;
      v32 = (_QWORD *)sub_22077B0(0x10u);
      if ( v32 )
      {
        v32[1] = a3;
        *v32 = &unk_4A09EA8;
      }
      v33 = v100[0];
      v100[0] = v32;
      if ( v33 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v33 + 8LL))(v33);
      v34 = v29;
      v36 = (*(__int64 (__fastcall **)(__int64 *))(*v27 + 24))(v27);
      if ( (v29[3] & 2) == 0 )
        v34 = (_QWORD *)*v29;
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v29[3] & 0xFFFFFFFFFFFFFFF8LL))(
        v34,
        v36,
        v35,
        v100);
      if ( v100[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v100[0] + 8LL))(v100[0]);
      v29 += 4;
    }
    while ( v31 != v29 );
    v4 = v94;
  }
  v37 = *(_DWORD *)(v4 + 56);
  v38 = v4 + 32;
  if ( !v37 )
  {
LABEL_73:
    ++*(_QWORD *)(v4 + 32);
    goto LABEL_74;
  }
LABEL_37:
  v39 = *(_QWORD *)(v4 + 40);
  v40 = 1;
  v41 = (v37 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v42 = (__int64 **)(v39 + 32LL * v41);
  v43 = 0;
  v44 = *v42;
  if ( a3 == *v42 )
  {
LABEL_38:
    v95 = (__int64)(v42 + 1);
    goto LABEL_39;
  }
  while ( v44 != (__int64 *)-4096LL )
  {
    if ( v44 == (__int64 *)-8192LL && !v43 )
      v43 = v42;
    v41 = (v37 - 1) & (v40 + v41);
    v42 = (__int64 **)(v39 + 32LL * v41);
    v44 = *v42;
    if ( a3 == *v42 )
      goto LABEL_38;
    ++v40;
  }
  v60 = *(_DWORD *)(v4 + 48);
  if ( !v43 )
    v43 = v42;
  ++*(_QWORD *)(v4 + 32);
  v61 = v60 + 1;
  if ( 4 * v61 >= 3 * v37 )
  {
LABEL_74:
    sub_22D0430(v38, 2 * v37);
    v62 = *(_DWORD *)(v4 + 56);
    if ( v62 )
    {
      v63 = v62 - 1;
      v64 = *(_QWORD *)(v4 + 40);
      v65 = (v62 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v61 = *(_DWORD *)(v4 + 48) + 1;
      v43 = (__int64 **)(v64 + 32LL * v65);
      v66 = *v43;
      if ( a3 != *v43 )
      {
        v67 = 1;
        v68 = 0;
        while ( v66 != (__int64 *)-4096LL )
        {
          if ( v66 == (__int64 *)-8192LL && !v68 )
            v68 = v43;
          v65 = v63 & (v67 + v65);
          v43 = (__int64 **)(v64 + 32LL * v65);
          v66 = *v43;
          if ( a3 == *v43 )
            goto LABEL_69;
          ++v67;
        }
        if ( v68 )
          v43 = v68;
      }
      goto LABEL_69;
    }
    goto LABEL_139;
  }
  if ( v37 - *(_DWORD *)(v4 + 52) - v61 <= v37 >> 3 )
  {
    sub_22D0430(v38, v37);
    v86 = *(_DWORD *)(v4 + 56);
    if ( v86 )
    {
      v87 = v86 - 1;
      v88 = *(_QWORD *)(v4 + 40);
      v89 = 0;
      v90 = (v86 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v91 = 1;
      v61 = *(_DWORD *)(v4 + 48) + 1;
      v43 = (__int64 **)(v88 + 32LL * v90);
      v92 = *v43;
      if ( a3 != *v43 )
      {
        while ( v92 != (__int64 *)-4096LL )
        {
          if ( !v89 && v92 == (__int64 *)-8192LL )
            v89 = v43;
          v90 = v87 & (v91 + v90);
          v43 = (__int64 **)(v88 + 32LL * v90);
          v92 = *v43;
          if ( a3 == *v43 )
            goto LABEL_69;
          ++v91;
        }
        if ( v89 )
          v43 = v89;
      }
      goto LABEL_69;
    }
LABEL_139:
    ++*(_DWORD *)(v4 + 48);
    BUG();
  }
LABEL_69:
  *(_DWORD *)(v4 + 48) = v61;
  if ( *v43 != (__int64 *)-4096LL )
    --*(_DWORD *)(v4 + 52);
  *v43 = a3;
  v95 = (__int64)(v43 + 1);
  v43[2] = (__int64 *)(v43 + 1);
  v43[1] = (__int64 *)(v43 + 1);
  v43[3] = 0;
LABEL_39:
  (*(void (__fastcall **)(_QWORD *, __int64 *, __int64 *, __int64, __int64))(*v27 + 16))(v100, v27, a3, v4, a4);
  v45 = (_QWORD *)sub_22077B0(0x20u);
  v45[2] = a2;
  v46 = v100[0];
  v100[0] = 0;
  v45[3] = v46;
  sub_2208C80(v45, v95);
  ++*(_QWORD *)(v95 + 16);
  if ( v100[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v100[0] + 8LL))(v100[0]);
  if ( v93 )
  {
    v47 = *(_QWORD **)(v93 + 864);
    if ( v47 != &v47[4 * *(unsigned int *)(v93 + 872)] )
    {
      v98 = v4;
      v48 = &v47[4 * *(unsigned int *)(v93 + 872)];
      do
      {
        v100[0] = 0;
        v49 = (_QWORD *)sub_22077B0(0x10u);
        if ( v49 )
        {
          v49[1] = a3;
          *v49 = &unk_4A09EA8;
        }
        v50 = v100[0];
        v100[0] = v49;
        if ( v50 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v50 + 8LL))(v50);
        v51 = v47;
        v53 = (*(__int64 (__fastcall **)(__int64 *))(*v27 + 24))(v27);
        if ( (v47[3] & 2) == 0 )
          v51 = (_QWORD *)*v47;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD *))(v47[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v51,
          v53,
          v52,
          v100);
        if ( v100[0] )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v100[0] + 8LL))(v100[0]);
        v47 += 4;
      }
      while ( v48 != v47 );
      v4 = v98;
    }
  }
  v54 = *(unsigned int *)(v4 + 88);
  v55 = *(_QWORD *)(v4 + 72);
  if ( (_DWORD)v54 )
  {
    v56 = 1;
    for ( m = (v54 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = (v54 - 1) & v59 )
    {
      v58 = (__int64 **)(v55 + 24LL * m);
      if ( *v58 == a2 && a3 == v58[1] )
        break;
      if ( *v58 == (__int64 *)-4096LL && v58[1] == (__int64 *)-4096LL )
        goto LABEL_82;
      v59 = v56 + m;
      ++v56;
    }
  }
  else
  {
LABEL_82:
    v58 = (__int64 **)(v55 + 24 * v54);
  }
  v18 = *(__int64 **)(v95 + 8);
  v58[2] = v18;
  return v18[3];
}
