// Function: sub_18DDD00
// Address: 0x18ddd00
//
__int64 __fastcall sub_18DDD00(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r13
  unsigned int v10; // esi
  __int64 v11; // r14
  __int64 v12; // rdx
  int v13; // r10d
  unsigned __int64 *v14; // rdi
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned int i; // r8d
  __int64 *v19; // rcx
  __int64 v20; // r11
  unsigned int v21; // r8d
  unsigned __int64 v22; // rax
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // rdx
  int v26; // r9d
  unsigned int v27; // edi
  unsigned __int64 *v28; // r8
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rsi
  unsigned int j; // eax
  unsigned __int64 v32; // rsi
  unsigned int v33; // eax
  unsigned int v34; // r15d
  int v36; // edx
  int v37; // ecx
  unsigned int v38; // eax
  unsigned int v39; // esi
  __int64 v40; // rdx
  int v41; // r10d
  unsigned __int64 *v42; // r9
  unsigned __int64 v43; // rcx
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rax
  unsigned int v46; // edi
  unsigned __int64 *v47; // rcx
  unsigned __int64 v48; // r11
  unsigned int v49; // edi
  int v50; // edx
  int v51; // edx
  __int64 v52; // rsi
  int v53; // r9d
  unsigned int k; // eax
  unsigned __int64 v55; // rcx
  unsigned int v56; // eax
  int v57; // ecx
  __int64 v58; // rdx
  int v59; // r9d
  unsigned __int64 *v60; // r8
  int v61; // edi
  unsigned __int64 v62; // rsi
  unsigned __int64 v63; // rsi
  unsigned int v64; // eax
  unsigned __int64 v65; // rsi
  unsigned int v66; // eax
  int v67; // edx
  int v68; // edi
  int v69; // edx
  int v70; // edx
  __int64 v71; // r8
  int v72; // r9d
  unsigned int m; // eax
  __int64 *v74; // rsi
  __int64 v75; // rdi
  unsigned int v76; // eax
  int v77; // [rsp+8h] [rbp-38h]
  int v78; // [rsp+8h] [rbp-38h]

  v7 = sub_18DD900(a2, a4, a1 + 40);
  v8 = sub_18DD900(a3, a4, a1 + 40);
  if ( v7 == v8 )
    return 1;
  v9 = v8;
  if ( v7 > v8 )
  {
    v10 = *(_DWORD *)(a1 + 32);
    v22 = v7;
    v11 = a1 + 8;
    v7 = v9;
    v9 = v22;
    if ( v10 )
      goto LABEL_4;
LABEL_13:
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_14;
  }
  v10 = *(_DWORD *)(a1 + 32);
  v11 = a1 + 8;
  if ( !v10 )
    goto LABEL_13;
LABEL_4:
  v12 = *(_QWORD *)(a1 + 16);
  v13 = 1;
  v14 = 0;
  v15 = (((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
         | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
        | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
  v16 = ((9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13)))) >> 15)
      ^ (9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13))));
  v17 = ((v16 - 1 - (v16 << 27)) >> 31) ^ (v16 - 1 - (v16 << 27));
  for ( i = v17 & (v10 - 1); ; i = (v10 - 1) & v21 )
  {
    v19 = (__int64 *)(v12 + 24LL * i);
    v20 = *v19;
    if ( v7 == *v19 && v9 == v19[1] )
      return *((unsigned __int8 *)v19 + 16);
    if ( v20 == -8 )
      break;
    if ( v20 == -16 && v19[1] == -16 && !v14 )
      v14 = (unsigned __int64 *)(v12 + 24LL * i);
LABEL_11:
    v21 = v13 + i;
    ++v13;
  }
  if ( v19[1] != -8 )
    goto LABEL_11;
  v36 = *(_DWORD *)(a1 + 24);
  if ( !v14 )
    v14 = (unsigned __int64 *)v19;
  ++*(_QWORD *)(a1 + 8);
  v37 = v36 + 1;
  if ( 4 * (v36 + 1) >= 3 * v10 )
  {
LABEL_14:
    sub_1350E40(v11, 2 * v10);
    v23 = *(_DWORD *)(a1 + 32);
    if ( v23 )
    {
      v24 = v23 - 1;
      v26 = 1;
      v27 = (unsigned int)v9 >> 9;
      v28 = 0;
      v29 = (((v27 ^ ((unsigned int)v9 >> 4)
             | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v27 ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
          ^ ((v27 ^ ((unsigned int)v9 >> 4)
            | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v27 ^ ((unsigned int)v9 >> 4)) << 32));
      v30 = ((9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13)))) >> 15)
          ^ (9 * (((v29 - 1 - (v29 << 13)) >> 8) ^ (v29 - 1 - (v29 << 13))));
      for ( j = v24 & (((v30 - 1 - (v30 << 27)) >> 31) ^ (v30 - 1 - ((_DWORD)v30 << 27))); ; j = v24 & v33 )
      {
        v25 = *(_QWORD *)(a1 + 16);
        v14 = (unsigned __int64 *)(v25 + 24LL * j);
        v32 = *v14;
        if ( v7 == *v14 && v9 == v14[1] )
          break;
        if ( v32 == -8 )
        {
          if ( v14[1] == -8 )
          {
LABEL_94:
            if ( v28 )
              v14 = v28;
            v37 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_32;
          }
        }
        else if ( v32 == -16 && v14[1] == -16 && !v28 )
        {
          v28 = (unsigned __int64 *)(v25 + 24LL * j);
        }
        v33 = v26 + j;
        ++v26;
      }
      goto LABEL_86;
    }
LABEL_107:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
  if ( v10 - *(_DWORD *)(a1 + 28) - v37 <= v10 >> 3 )
  {
    v77 = v17;
    sub_1350E40(v11, v10);
    v50 = *(_DWORD *)(a1 + 32);
    if ( v50 )
    {
      v51 = v50 - 1;
      v28 = 0;
      v53 = 1;
      for ( k = v51 & v77; ; k = v51 & v56 )
      {
        v52 = *(_QWORD *)(a1 + 16);
        v14 = (unsigned __int64 *)(v52 + 24LL * k);
        v55 = *v14;
        if ( v7 == *v14 && v9 == v14[1] )
          break;
        if ( v55 == -8 )
        {
          if ( v14[1] == -8 )
            goto LABEL_94;
        }
        else if ( v55 == -16 && v14[1] == -16 && !v28 )
        {
          v28 = (unsigned __int64 *)(v52 + 24LL * k);
        }
        v56 = v53 + k;
        ++v53;
      }
LABEL_86:
      v37 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_32;
    }
    goto LABEL_107;
  }
LABEL_32:
  *(_DWORD *)(a1 + 24) = v37;
  if ( *v14 != -8 || v14[1] != -8 )
    --*(_DWORD *)(a1 + 28);
  *v14 = v7;
  v14[1] = v9;
  *((_BYTE *)v14 + 16) = 1;
  v38 = sub_18DE770(a1, v7, v9, a4);
  v39 = *(_DWORD *)(a1 + 32);
  v34 = v38;
  if ( !v39 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_57;
  }
  v40 = *(_QWORD *)(a1 + 16);
  v41 = 1;
  v42 = 0;
  v43 = (((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
         | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
        | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
  v44 = ((9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13)))) >> 15)
      ^ (9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13))));
  v45 = ((v44 - 1 - (v44 << 27)) >> 31) ^ (v44 - 1 - (v44 << 27));
  v46 = v45 & (v39 - 1);
  while ( 2 )
  {
    v47 = (unsigned __int64 *)(v40 + 24LL * v46);
    v48 = *v47;
    if ( v7 == *v47 && v9 == v47[1] )
      goto LABEL_46;
    if ( v48 != -8 )
    {
      if ( v48 == -16 && v47[1] == -16 && !v42 )
        v42 = (unsigned __int64 *)(v40 + 24LL * v46);
      goto LABEL_42;
    }
    if ( v47[1] != -8 )
    {
LABEL_42:
      v49 = v41 + v46;
      ++v41;
      v46 = (v39 - 1) & v49;
      continue;
    }
    break;
  }
  v67 = *(_DWORD *)(a1 + 24);
  if ( v42 )
    v47 = v42;
  ++*(_QWORD *)(a1 + 8);
  v68 = v67 + 1;
  if ( 4 * (v67 + 1) < 3 * v39 )
  {
    if ( v39 - *(_DWORD *)(a1 + 28) - v68 > v39 >> 3 )
      goto LABEL_71;
    v78 = v45;
    sub_1350E40(v11, v39);
    v69 = *(_DWORD *)(a1 + 32);
    if ( v69 )
    {
      v70 = v69 - 1;
      v47 = 0;
      v72 = 1;
      for ( m = v70 & v78; ; m = v70 & v76 )
      {
        v71 = *(_QWORD *)(a1 + 16);
        v74 = (__int64 *)(v71 + 24LL * m);
        v75 = *v74;
        if ( v7 == *v74 && v9 == v74[1] )
        {
          v47 = (unsigned __int64 *)(v71 + 24LL * m);
          v68 = *(_DWORD *)(a1 + 24) + 1;
          goto LABEL_71;
        }
        if ( v75 == -8 )
        {
          if ( v74[1] == -8 )
          {
            if ( !v47 )
              v47 = (unsigned __int64 *)(v71 + 24LL * m);
            v68 = *(_DWORD *)(a1 + 24) + 1;
            goto LABEL_71;
          }
        }
        else if ( v75 == -16 && v74[1] == -16 && !v47 )
        {
          v47 = (unsigned __int64 *)(v71 + 24LL * m);
        }
        v76 = v72 + m;
        ++v72;
      }
    }
LABEL_108:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_57:
  sub_1350E40(v11, 2 * v39);
  v57 = *(_DWORD *)(a1 + 32);
  if ( !v57 )
    goto LABEL_108;
  v59 = 1;
  v60 = 0;
  v61 = v57 - 1;
  v62 = (((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
         | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
        | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32));
  v63 = ((9 * (((v62 - 1 - (v62 << 13)) >> 8) ^ (v62 - 1 - (v62 << 13)))) >> 15)
      ^ (9 * (((v62 - 1 - (v62 << 13)) >> 8) ^ (v62 - 1 - (v62 << 13))));
  v64 = (v57 - 1) & (((v63 - 1 - (v63 << 27)) >> 31) ^ (v63 - 1 - ((_DWORD)v63 << 27)));
  while ( 2 )
  {
    v58 = *(_QWORD *)(a1 + 16);
    v47 = (unsigned __int64 *)(v58 + 24LL * v64);
    v65 = *v47;
    if ( v7 == *v47 && v9 == v47[1] )
    {
      v68 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_71;
    }
    if ( v65 != -8 )
    {
      if ( v65 == -16 && v47[1] == -16 && !v60 )
        v60 = (unsigned __int64 *)(v58 + 24LL * v64);
      goto LABEL_65;
    }
    if ( v47[1] != -8 )
    {
LABEL_65:
      v66 = v59 + v64;
      ++v59;
      v64 = v61 & v66;
      continue;
    }
    break;
  }
  if ( v60 )
    v47 = v60;
  v68 = *(_DWORD *)(a1 + 24) + 1;
LABEL_71:
  *(_DWORD *)(a1 + 24) = v68;
  if ( *v47 != -8 || v47[1] != -8 )
    --*(_DWORD *)(a1 + 28);
  *v47 = v7;
  v47[1] = v9;
  *((_BYTE *)v47 + 16) = 0;
LABEL_46:
  *((_BYTE *)v47 + 16) = v34;
  return v34;
}
