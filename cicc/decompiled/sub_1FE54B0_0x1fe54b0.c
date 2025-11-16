// Function: sub_1FE54B0
// Address: 0x1fe54b0
//
__int64 __fastcall sub_1FE54B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned int v8; // edi
  int v9; // r9d
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rsi
  unsigned int i; // eax
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 (__fastcall *v18)(__int64, unsigned __int8); // r15
  unsigned int v19; // eax
  __int64 v20; // r8
  int v21; // r9d
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned int v24; // esi
  __int64 v25; // rcx
  int v26; // r11d
  __int64 *v27; // r10
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdx
  unsigned int j; // r8d
  __int64 *v32; // rdi
  __int64 v33; // r15
  unsigned int v34; // r8d
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 v37; // rcx
  __int64 *v38; // r11
  int v39; // r14d
  unsigned __int64 v40; // r8
  unsigned __int64 v41; // r8
  unsigned __int64 v42; // rdx
  unsigned int v43; // r9d
  __int64 *v44; // r8
  __int64 v45; // r15
  unsigned int v46; // r9d
  int v47; // esi
  int v48; // esi
  __int64 v49; // rcx
  unsigned int v50; // r8d
  int v51; // r9d
  __int64 *v52; // r10
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 v56; // rdi
  unsigned int v57; // edx
  int v58; // esi
  int v59; // esi
  __int64 v60; // rcx
  unsigned int v61; // r8d
  int v62; // r10d
  __int64 *v63; // r9
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  unsigned int v66; // edx
  __int64 v67; // r8
  unsigned int v68; // edx
  int v69; // ecx
  int v70; // r8d
  int v71; // ecx
  int v72; // esi
  __int64 v73; // r9
  int v74; // r10d
  unsigned int k; // edx
  __int64 *v76; // rcx
  __int64 v77; // r8
  unsigned int v78; // edx
  int v79; // ecx
  int v80; // r9d
  int v81; // ecx
  int v82; // esi
  __int64 v83; // r9
  int v84; // r10d
  unsigned int m; // edx
  __int64 *v86; // rcx
  __int64 v87; // rdi
  unsigned int v88; // edx
  int v89; // [rsp+0h] [rbp-40h]
  int v90; // [rsp+0h] [rbp-40h]
  unsigned int v91; // [rsp+Ch] [rbp-34h]
  unsigned int v92; // [rsp+Ch] [rbp-34h]
  unsigned int v93; // [rsp+Ch] [rbp-34h]
  unsigned int v94; // [rsp+Ch] [rbp-34h]

  v6 = *(unsigned int *)(a1 + 104);
  if ( (_DWORD)v6 )
  {
    v7 = *(_QWORD *)(a1 + 88);
    v8 = (unsigned int)a3 >> 9;
    v9 = 1;
    v10 = (((v8 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v8 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
        ^ ((v8 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v8 ^ ((unsigned int)a3 >> 4)) << 32));
    v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
        ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
    for ( i = (v6 - 1) & (((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - ((_DWORD)v11 << 27))); ; i = (v6 - 1) & v14 )
    {
      v13 = v7 + 24LL * i;
      if ( a2 == *(_QWORD *)v13 && a3 == *(_QWORD *)(v13 + 8) )
        break;
      if ( *(_QWORD *)v13 == -8 && *(_QWORD *)(v13 + 8) == -8 )
        goto LABEL_10;
      v14 = v9 + i;
      ++v9;
    }
    if ( v13 != v7 + 24 * v6 )
      return *(unsigned int *)(v13 + 16);
  }
LABEL_10:
  v16 = sub_1E0A0C0(*(_QWORD *)(a1 + 8));
  v17 = *(_QWORD *)(a1 + 16);
  v18 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v17 + 288LL);
  v19 = 8 * sub_15A9520(v16, 0);
  if ( v19 == 32 )
  {
    v22 = 5;
  }
  else if ( v19 > 0x20 )
  {
    v22 = 6;
    if ( v19 != 64 )
    {
      v22 = 0;
      if ( v19 == 128 )
        v22 = 7;
    }
  }
  else
  {
    v22 = 3;
    if ( v19 != 8 )
    {
      LOBYTE(v22) = v19 == 16;
      v22 = (unsigned int)(4 * v22);
    }
  }
  if ( v18 == sub_1D45FB0 )
    v23 = *(_QWORD *)(v17 + 8 * (v22 & 7) + 120);
  else
    v23 = v18(v17, v22);
  result = sub_1E6B9A0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL), v23, (unsigned __int8 *)byte_3F871B3, 0, v20, v21);
  v24 = *(_DWORD *)(a1 + 104);
  if ( !v24 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_54;
  }
  v25 = *(_QWORD *)(a1 + 88);
  v26 = 1;
  v27 = 0;
  v28 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v29 = ((9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13)))) >> 15)
      ^ (9 * (((v28 - 1 - (v28 << 13)) >> 8) ^ (v28 - 1 - (v28 << 13))));
  v30 = ((v29 - 1 - (v29 << 27)) >> 31) ^ (v29 - 1 - (v29 << 27));
  for ( j = v30 & (v24 - 1); ; j = (v24 - 1) & v34 )
  {
    v32 = (__int64 *)(v25 + 24LL * j);
    v33 = *v32;
    if ( a2 == *v32 && a3 == v32[1] )
      goto LABEL_30;
    if ( v33 == -8 )
      break;
    if ( v33 == -16 && v32[1] == -16 && !v27 )
      v27 = (__int64 *)(v25 + 24LL * j);
LABEL_24:
    v34 = v26 + j;
    ++v26;
  }
  if ( v32[1] != -8 )
    goto LABEL_24;
  v69 = *(_DWORD *)(a1 + 96);
  if ( v27 )
    v32 = v27;
  ++*(_QWORD *)(a1 + 80);
  v70 = v69 + 1;
  if ( 4 * (v69 + 1) < 3 * v24 )
  {
    if ( v24 - *(_DWORD *)(a1 + 100) - v70 > v24 >> 3 )
      goto LABEL_68;
    v89 = v30;
    v93 = result;
    sub_1FE4EE0(a1 + 80, v24);
    v71 = *(_DWORD *)(a1 + 104);
    if ( v71 )
    {
      v72 = v71 - 1;
      v73 = *(_QWORD *)(a1 + 88);
      v32 = 0;
      result = v93;
      v74 = 1;
      for ( k = (v71 - 1) & v89; ; k = v72 & v78 )
      {
        v76 = (__int64 *)(v73 + 24LL * k);
        v77 = *v76;
        if ( a2 == *v76 && a3 == v76[1] )
        {
          v70 = *(_DWORD *)(a1 + 96) + 1;
          v32 = (__int64 *)(v73 + 24LL * k);
          goto LABEL_68;
        }
        if ( v77 == -8 )
        {
          if ( v76[1] == -8 )
          {
            if ( !v32 )
              v32 = (__int64 *)(v73 + 24LL * k);
            v70 = *(_DWORD *)(a1 + 96) + 1;
            goto LABEL_68;
          }
        }
        else if ( v77 == -16 && v76[1] == -16 && !v32 )
        {
          v32 = (__int64 *)(v73 + 24LL * k);
        }
        v78 = v74 + k;
        ++v74;
      }
    }
LABEL_125:
    ++*(_DWORD *)(a1 + 96);
    BUG();
  }
LABEL_54:
  v92 = result;
  sub_1FE4EE0(a1 + 80, 2 * v24);
  v58 = *(_DWORD *)(a1 + 104);
  if ( !v58 )
    goto LABEL_125;
  v59 = v58 - 1;
  result = v92;
  v61 = (unsigned int)a3 >> 9;
  v62 = 1;
  v63 = 0;
  v64 = (((v61 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v61 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((v61 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v61 ^ ((unsigned int)a3 >> 4)) << 32));
  v65 = ((9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13)))) >> 15)
      ^ (9 * (((v64 - 1 - (v64 << 13)) >> 8) ^ (v64 - 1 - (v64 << 13))));
  v66 = v59 & (((v65 - 1 - (v65 << 27)) >> 31) ^ (v65 - 1 - ((_DWORD)v65 << 27)));
  while ( 2 )
  {
    v60 = *(_QWORD *)(a1 + 88);
    v32 = (__int64 *)(v60 + 24LL * v66);
    v67 = *v32;
    if ( a2 == *v32 && a3 == v32[1] )
    {
      v70 = *(_DWORD *)(a1 + 96) + 1;
      goto LABEL_68;
    }
    if ( v67 != -8 )
    {
      if ( v67 == -16 && v32[1] == -16 && !v63 )
        v63 = (__int64 *)(v60 + 24LL * v66);
      goto LABEL_62;
    }
    if ( v32[1] != -8 )
    {
LABEL_62:
      v68 = v62 + v66;
      ++v62;
      v66 = v59 & v68;
      continue;
    }
    break;
  }
  if ( v63 )
    v32 = v63;
  v70 = *(_DWORD *)(a1 + 96) + 1;
LABEL_68:
  *(_DWORD *)(a1 + 96) = v70;
  if ( *v32 != -8 || v32[1] != -8 )
    --*(_DWORD *)(a1 + 100);
  *v32 = a2;
  v32[1] = a3;
  *((_DWORD *)v32 + 4) = 0;
LABEL_30:
  *((_DWORD *)v32 + 4) = result;
  v35 = *(_DWORD *)(a1 + 136);
  v36 = a1 + 112;
  if ( !v35 )
  {
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_44;
  }
  v37 = *(_QWORD *)(a1 + 120);
  v38 = 0;
  v39 = 1;
  v40 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v41 = ((9 * (((v40 - 1 - (v40 << 13)) >> 8) ^ (v40 - 1 - (v40 << 13)))) >> 15)
      ^ (9 * (((v40 - 1 - (v40 << 13)) >> 8) ^ (v40 - 1 - (v40 << 13))));
  v42 = ((v41 - 1 - (v41 << 27)) >> 31) ^ (v41 - 1 - (v41 << 27));
  v43 = v42 & (v35 - 1);
  while ( 2 )
  {
    v44 = (__int64 *)(v37 + 24LL * v43);
    v45 = *v44;
    if ( a2 == *v44 && a3 == v44[1] )
      goto LABEL_40;
    if ( v45 != -8 )
    {
      if ( v45 == -16 && v44[1] == -16 && !v38 )
        v38 = (__int64 *)(v37 + 24LL * v43);
      goto LABEL_38;
    }
    if ( v44[1] != -8 )
    {
LABEL_38:
      v46 = v39 + v43;
      ++v39;
      v43 = (v35 - 1) & v46;
      continue;
    }
    break;
  }
  v79 = *(_DWORD *)(a1 + 128);
  if ( v38 )
    v44 = v38;
  ++*(_QWORD *)(a1 + 112);
  v80 = v79 + 1;
  if ( 4 * (v79 + 1) < 3 * v35 )
  {
    if ( v35 - *(_DWORD *)(a1 + 132) - v80 > v35 >> 3 )
      goto LABEL_87;
    v90 = v42;
    v94 = result;
    sub_1FE4EE0(v36, v35);
    v81 = *(_DWORD *)(a1 + 136);
    if ( v81 )
    {
      v82 = v81 - 1;
      v44 = 0;
      result = v94;
      v84 = 1;
      for ( m = (v81 - 1) & v90; ; m = v82 & v88 )
      {
        v83 = *(_QWORD *)(a1 + 120);
        v86 = (__int64 *)(v83 + 24LL * m);
        v87 = *v86;
        if ( a2 == *v86 && a3 == v86[1] )
        {
          v44 = (__int64 *)(v83 + 24LL * m);
          v80 = *(_DWORD *)(a1 + 128) + 1;
          goto LABEL_87;
        }
        if ( v87 == -8 )
        {
          if ( v86[1] == -8 )
          {
            if ( !v44 )
              v44 = (__int64 *)(v83 + 24LL * m);
            v80 = *(_DWORD *)(a1 + 128) + 1;
            goto LABEL_87;
          }
        }
        else if ( v87 == -16 && v86[1] == -16 && !v44 )
        {
          v44 = (__int64 *)(v83 + 24LL * m);
        }
        v88 = v84 + m;
        ++v84;
      }
    }
LABEL_126:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_44:
  v91 = result;
  sub_1FE4EE0(v36, 2 * v35);
  v47 = *(_DWORD *)(a1 + 136);
  if ( !v47 )
    goto LABEL_126;
  v48 = v47 - 1;
  result = v91;
  v50 = (unsigned int)a3 >> 9;
  v51 = 1;
  v52 = 0;
  v53 = (((v50 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v50 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((v50 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v50 ^ ((unsigned int)a3 >> 4)) << 32));
  v54 = ((9 * (((v53 - 1 - (v53 << 13)) >> 8) ^ (v53 - 1 - (v53 << 13)))) >> 15)
      ^ (9 * (((v53 - 1 - (v53 << 13)) >> 8) ^ (v53 - 1 - (v53 << 13))));
  v55 = v48 & (((v54 - 1 - (v54 << 27)) >> 31) ^ (v54 - 1 - ((_DWORD)v54 << 27)));
  while ( 2 )
  {
    v49 = *(_QWORD *)(a1 + 120);
    v44 = (__int64 *)(v49 + 24LL * v55);
    v56 = *v44;
    if ( a2 == *v44 && a3 == v44[1] )
    {
      v80 = *(_DWORD *)(a1 + 128) + 1;
      goto LABEL_87;
    }
    if ( v56 != -8 )
    {
      if ( v56 == -16 && v44[1] == -16 && !v52 )
        v52 = (__int64 *)(v49 + 24LL * v55);
      goto LABEL_52;
    }
    if ( v44[1] != -8 )
    {
LABEL_52:
      v57 = v51 + v55;
      ++v51;
      v55 = v48 & v57;
      continue;
    }
    break;
  }
  if ( v52 )
    v44 = v52;
  v80 = *(_DWORD *)(a1 + 128) + 1;
LABEL_87:
  *(_DWORD *)(a1 + 128) = v80;
  if ( *v44 != -8 || v44[1] != -8 )
    --*(_DWORD *)(a1 + 132);
  *v44 = a2;
  v44[1] = a3;
  *((_DWORD *)v44 + 4) = 0;
LABEL_40:
  *((_DWORD *)v44 + 4) = result;
  return result;
}
