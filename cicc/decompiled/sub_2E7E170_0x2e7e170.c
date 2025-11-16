// Function: sub_2E7E170
// Address: 0x2e7e170
//
void __fastcall sub_2E7E170(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // edi
  unsigned int v19; // eax
  __int64 *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r14
  int v23; // r15d
  int v24; // r10d
  unsigned int v25; // r13d
  _QWORD *v26; // rdx
  unsigned int v27; // r9d
  _QWORD *v28; // rax
  __int64 v29; // r8
  _QWORD *v30; // rax
  char *v31; // rdi
  size_t v32; // rdx
  int v33; // edx
  int v34; // r9d
  int v35; // eax
  int v36; // ecx
  int v37; // eax
  int v38; // eax
  int v39; // eax
  int v40; // edi
  __int64 v41; // rsi
  int v42; // r10d
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  __int64 v46; // r13
  __int64 v47; // rsi
  int v48; // r9d
  _QWORD *v49; // r8
  int v50; // eax
  int v51; // esi
  int v52; // r10d
  __int64 v53; // rdi
  int v54; // eax
  int v55; // eax
  int v56; // r9d
  __int64 v57; // rdi
  __int64 v58; // r13
  __int64 v59; // rsi
  int v60; // [rsp+0h] [rbp-80h]
  int v61; // [rsp+8h] [rbp-78h]
  __int64 v62; // [rsp+8h] [rbp-78h]
  int v63; // [rsp+8h] [rbp-78h]
  unsigned int v64; // [rsp+8h] [rbp-78h]
  char *v65; // [rsp+10h] [rbp-70h] BYREF
  __int64 v66; // [rsp+18h] [rbp-68h]
  _BYTE dest[16]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v68; // [rsp+30h] [rbp-50h] BYREF
  __int64 v69; // [rsp+40h] [rbp-40h]

  v3 = a2;
  if ( !(unsigned __int8)sub_2E88ED0(a3, 0, a3) )
  {
    sub_2E79700(a1, a2);
    return;
  }
  if ( *(_WORD *)(a2 + 68) == 21 )
    v3 = sub_2E78040(a2);
  sub_2E79610(&v68, a1, v3);
  v6 = v69;
  v7 = *(_QWORD *)(a1 + 696);
  v8 = *(unsigned int *)(a1 + 712);
  if ( v69 != v7 + 32 * v8 )
  {
    v9 = *(unsigned int *)(v69 + 16);
    v65 = dest;
    v66 = 0x100000000LL;
    if ( (_DWORD)v9 && &v65 != (char **)(v69 + 8) )
    {
      v31 = dest;
      v32 = 8;
      if ( (_DWORD)v9 == 1
        || (v60 = v9,
            v62 = v69,
            sub_C8D5F0((__int64)&v65, dest, (unsigned int)v9, 8u, v69, v9),
            v6 = v62,
            v31 = v65,
            LODWORD(v9) = v60,
            (v32 = 8LL * *(unsigned int *)(v62 + 16)) != 0) )
      {
        v61 = v9;
        memcpy(v31, *(const void **)(v6 + 8), v32);
        LODWORD(v9) = v61;
      }
      LODWORD(v66) = v9;
      v7 = *(_QWORD *)(a1 + 696);
      LODWORD(v8) = *(_DWORD *)(a1 + 712);
    }
    if ( (_DWORD)v8 )
    {
      v10 = (unsigned int)a3 >> 9;
      v11 = (unsigned int)v10 ^ ((unsigned int)a3 >> 4);
      v12 = ((_DWORD)v8 - 1) & ((unsigned int)v10 ^ ((unsigned int)a3 >> 4));
      v13 = (__int64 *)(v7 + 32 * v12);
      v14 = *v13;
      if ( *v13 == a3 )
      {
LABEL_8:
        v15 = (__int64)(v13 + 1);
LABEL_9:
        sub_2E78490(v15, &v65, v10, v11, v14, v12);
        if ( v65 != dest )
          _libc_free((unsigned __int64)v65);
        goto LABEL_11;
      }
      v63 = 1;
      v10 = 0;
      while ( v14 != -4096 )
      {
        if ( !v10 && v14 == -8192 )
          v10 = (__int64)v13;
        v12 = ((_DWORD)v8 - 1) & (unsigned int)(v63 + v12);
        v13 = (__int64 *)(v7 + 32LL * (unsigned int)v12);
        v14 = *v13;
        if ( *v13 == a3 )
          goto LABEL_8;
        ++v63;
      }
      v37 = *(_DWORD *)(a1 + 704);
      if ( !v10 )
        v10 = (__int64)v13;
      ++*(_QWORD *)(a1 + 688);
      v38 = v37 + 1;
      if ( 4 * v38 < (unsigned int)(3 * v8) )
      {
        v14 = (unsigned int)(v8 - (v38 + *(_DWORD *)(a1 + 708)));
        if ( (unsigned int)v14 > (unsigned int)v8 >> 3 )
        {
LABEL_48:
          *(_DWORD *)(a1 + 704) = v38;
          if ( *(_QWORD *)v10 != -4096 )
            --*(_DWORD *)(a1 + 708);
          *(_QWORD *)v10 = a3;
          v15 = v10 + 8;
          *(_QWORD *)(v10 + 8) = v10 + 24;
          *(_QWORD *)(v10 + 16) = 0x100000000LL;
          goto LABEL_9;
        }
        v64 = v11;
        sub_2E7DD40(a1 + 688, v8);
        v50 = *(_DWORD *)(a1 + 712);
        if ( v50 )
        {
          v51 = v50 - 1;
          v52 = 1;
          v12 = 0;
          v14 = *(_QWORD *)(a1 + 696);
          v11 = (v50 - 1) & v64;
          v38 = *(_DWORD *)(a1 + 704) + 1;
          v10 = v14 + 32 * v11;
          v53 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 == a3 )
            goto LABEL_48;
          while ( v53 != -4096 )
          {
            if ( !v12 && v53 == -8192 )
              v12 = v10;
            v11 = v51 & (unsigned int)(v52 + v11);
            v10 = v14 + 32LL * (unsigned int)v11;
            v53 = *(_QWORD *)v10;
            if ( *(_QWORD *)v10 == a3 )
              goto LABEL_48;
            ++v52;
          }
          goto LABEL_56;
        }
        goto LABEL_98;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 688);
    }
    sub_2E7DD40(a1 + 688, 2 * v8);
    v39 = *(_DWORD *)(a1 + 712);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 696);
      v11 = (v39 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v38 = *(_DWORD *)(a1 + 704) + 1;
      v10 = v41 + 32 * v11;
      v14 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 == a3 )
        goto LABEL_48;
      v42 = 1;
      v12 = 0;
      while ( v14 != -4096 )
      {
        if ( !v12 && v14 == -8192 )
          v12 = v10;
        v11 = v40 & (unsigned int)(v42 + v11);
        v10 = v41 + 32LL * (unsigned int)v11;
        v14 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == a3 )
          goto LABEL_48;
        ++v42;
      }
LABEL_56:
      if ( v12 )
        v10 = v12;
      goto LABEL_48;
    }
LABEL_98:
    ++*(_DWORD *)(a1 + 704);
    BUG();
  }
LABEL_11:
  v16 = *(_DWORD *)(a1 + 744);
  v17 = *(_QWORD *)(a1 + 728);
  if ( !v16 )
    return;
  v18 = v16 - 1;
  v19 = (v16 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v20 = (__int64 *)(v17 + 24LL * v19);
  v21 = *v20;
  if ( v3 == *v20 )
  {
LABEL_13:
    if ( v20 == (__int64 *)(v17 + 24LL * v16) )
      return;
    v22 = v20[1];
    v23 = *((_DWORD *)v20 + 4);
    v24 = 1;
    v25 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
    v26 = 0;
    v27 = v25 & v18;
    v28 = (_QWORD *)(v17 + 24LL * (v25 & v18));
    v29 = *v28;
    if ( *v28 == a3 )
    {
LABEL_15:
      v30 = v28 + 1;
LABEL_16:
      *v30 = v22;
      *((_DWORD *)v30 + 2) = v23;
      return;
    }
    while ( v29 != -4096 )
    {
      if ( v29 == -8192 && !v26 )
        v26 = v28;
      v27 = v18 & (v24 + v27);
      v28 = (_QWORD *)(v17 + 24LL * v27);
      v29 = *v28;
      if ( *v28 == a3 )
        goto LABEL_15;
      ++v24;
    }
    if ( !v26 )
      v26 = v28;
    v35 = *(_DWORD *)(a1 + 736);
    ++*(_QWORD *)(a1 + 720);
    v36 = v35 + 1;
    if ( 4 * (v35 + 1) >= 3 * v16 )
    {
      sub_2E7DF70(a1 + 720, 2 * v16);
      v43 = *(_DWORD *)(a1 + 744);
      if ( v43 )
      {
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 728);
        LODWORD(v46) = v44 & v25;
        v36 = *(_DWORD *)(a1 + 736) + 1;
        v26 = (_QWORD *)(v45 + 24LL * (unsigned int)v46);
        v47 = *v26;
        if ( *v26 == a3 )
          goto LABEL_37;
        v48 = 1;
        v49 = 0;
        while ( v47 != -4096 )
        {
          if ( v47 == -8192 && !v49 )
            v49 = v26;
          v46 = v44 & (unsigned int)(v46 + v48);
          v26 = (_QWORD *)(v45 + 24 * v46);
          v47 = *v26;
          if ( *v26 == a3 )
            goto LABEL_37;
          ++v48;
        }
LABEL_63:
        if ( v49 )
          v26 = v49;
        goto LABEL_37;
      }
    }
    else
    {
      if ( v16 - *(_DWORD *)(a1 + 740) - v36 > v16 >> 3 )
      {
LABEL_37:
        *(_DWORD *)(a1 + 736) = v36;
        if ( *v26 != -4096 )
          --*(_DWORD *)(a1 + 740);
        *v26 = a3;
        v30 = v26 + 1;
        v26[1] = 0;
        *((_DWORD *)v26 + 4) = 0;
        goto LABEL_16;
      }
      sub_2E7DF70(a1 + 720, v16);
      v54 = *(_DWORD *)(a1 + 744);
      if ( v54 )
      {
        v55 = v54 - 1;
        v56 = 1;
        v49 = 0;
        v57 = *(_QWORD *)(a1 + 728);
        LODWORD(v58) = v55 & v25;
        v36 = *(_DWORD *)(a1 + 736) + 1;
        v26 = (_QWORD *)(v57 + 24LL * (unsigned int)v58);
        v59 = *v26;
        if ( *v26 == a3 )
          goto LABEL_37;
        while ( v59 != -4096 )
        {
          if ( v59 == -8192 && !v49 )
            v49 = v26;
          v58 = v55 & (unsigned int)(v58 + v56);
          v26 = (_QWORD *)(v57 + 24 * v58);
          v59 = *v26;
          if ( *v26 == a3 )
            goto LABEL_37;
          ++v56;
        }
        goto LABEL_63;
      }
    }
    ++*(_DWORD *)(a1 + 736);
    BUG();
  }
  v33 = 1;
  while ( v21 != -4096 )
  {
    v34 = v33 + 1;
    v19 = v18 & (v33 + v19);
    v20 = (__int64 *)(v17 + 24LL * v19);
    v21 = *v20;
    if ( v3 == *v20 )
      goto LABEL_13;
    v33 = v34;
  }
}
