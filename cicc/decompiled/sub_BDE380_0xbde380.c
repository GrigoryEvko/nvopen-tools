// Function: sub_BDE380
// Address: 0xbde380
//
void __fastcall sub_BDE380(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // r12
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r14
  unsigned __int64 v7; // rax
  _QWORD *v8; // rax
  _BYTE *v9; // rbx
  char v10; // al
  __int64 v11; // r12
  _BYTE *v12; // rax
  __int64 v13; // rbx
  _BYTE *v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r8
  char *v18; // rsi
  _BYTE *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // r12
  unsigned __int64 v23; // rcx
  unsigned int v24; // r9d
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // r15
  unsigned __int64 *v31; // rdx
  __int64 v32; // rax
  unsigned __int64 *v33; // r14
  __int64 v34; // rax
  unsigned __int64 *v35; // r13
  unsigned __int64 v36; // rcx
  unsigned __int64 *i; // rcx
  unsigned __int128 v38; // rdi
  unsigned __int64 *j; // rax
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rdx
  _QWORD *v42; // rcx
  int v43; // edx
  unsigned __int64 v44; // rdi
  unsigned __int64 *k; // rax
  unsigned __int64 v46; // r8
  __int64 v47; // r13
  unsigned __int64 *v48; // rbx
  _BYTE *v49; // rsi
  unsigned __int64 *v50; // r9
  _BYTE *v51; // rsi
  __int64 v52; // rax
  _QWORD *v53; // rcx
  __int64 v54; // rdi
  _BYTE *v55; // rax
  __int64 v56; // rdi
  _BYTE *v57; // rax
  __int64 v58; // [rsp+0h] [rbp-180h]
  __int64 v59; // [rsp+8h] [rbp-178h]
  unsigned __int64 v60; // [rsp+10h] [rbp-170h]
  __int64 v61; // [rsp+18h] [rbp-168h]
  _QWORD *v62; // [rsp+18h] [rbp-168h]
  unsigned __int64 *v64; // [rsp+30h] [rbp-150h]
  unsigned __int64 *v65; // [rsp+30h] [rbp-150h]
  unsigned __int64 *v66; // [rsp+30h] [rbp-150h]
  int v67; // [rsp+38h] [rbp-148h]
  _QWORD v68[4]; // [rsp+40h] [rbp-140h] BYREF
  char v69; // [rsp+60h] [rbp-120h]
  char v70; // [rsp+61h] [rbp-11Fh]
  void *base; // [rsp+70h] [rbp-110h] BYREF
  __int64 v72; // [rsp+78h] [rbp-108h]
  _BYTE v73[64]; // [rsp+80h] [rbp-100h] BYREF
  char *v74; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+C8h] [rbp-B8h]
  _BYTE v76[16]; // [rsp+D0h] [rbp-B0h] BYREF
  char v77; // [rsp+E0h] [rbp-A0h]
  char v78; // [rsp+E1h] [rbp-9Fh]

  v2 = (__int64 *)a1;
  v3 = a2;
  ++*(_QWORD *)(a1 + 288);
  if ( !*(_BYTE *)(a1 + 316) )
  {
    v4 = 4 * (*(_DWORD *)(a1 + 308) - *(_DWORD *)(a1 + 312));
    v5 = *(unsigned int *)(a1 + 304);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( v4 < (unsigned int)v5 )
    {
      sub_C8C990(a1 + 288);
      goto LABEL_7;
    }
    memset(*(void **)(a1 + 296), -1, 8 * v5);
  }
  *(_QWORD *)(a1 + 308) = 0;
LABEL_7:
  v6 = (_QWORD *)(a2 + 48);
  sub_E34890(a1 + 2072, a2);
  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v7 )
    goto LABEL_110;
  if ( !v7 )
    goto LABEL_24;
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
  {
LABEL_110:
    v78 = 1;
    v74 = "Basic Block does not have terminator!";
    v77 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)&v74);
    if ( *(_QWORD *)a1 )
LABEL_111:
      sub_BDBD80((__int64)v2, (_BYTE *)v3);
    return;
  }
  v8 = *(_QWORD **)(a2 + 56);
  if ( !v8 )
LABEL_24:
    BUG();
  if ( *((_BYTE *)v8 - 24) != 84 )
    goto LABEL_16;
  v13 = *(_QWORD *)(a2 + 16);
  if ( v13 )
  {
    while ( 1 )
    {
      v14 = *(_BYTE **)(v13 + 24);
      if ( (unsigned __int8)(*v14 - 30) <= 0xAu )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
        goto LABEL_67;
    }
    v15 = 0;
    base = v73;
    v72 = 0x800000000LL;
    v16 = v13;
    while ( 1 )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v16 + 24) - 30) <= 0xAu )
      {
        v16 = *(_QWORD *)(v16 + 8);
        ++v15;
        if ( !v16 )
          goto LABEL_35;
      }
    }
LABEL_35:
    v17 = v15 + 1;
    v18 = v73;
    if ( v17 > 8 )
    {
      v67 = v17;
      sub_C8D5F0(&base, v73, v17, 8);
      v14 = *(_BYTE **)(v13 + 24);
      LODWORD(v17) = v67;
      v18 = (char *)base + 8 * (unsigned int)v72;
    }
    v19 = v14;
LABEL_40:
    if ( v18 )
      *(_QWORD *)v18 = *((_QWORD *)v19 + 5);
    while ( 1 )
    {
      v13 = *(_QWORD *)(v13 + 8);
      if ( !v13 )
        break;
      v19 = *(_BYTE **)(v13 + 24);
      if ( (unsigned __int8)(*v19 - 30) <= 0xAu )
      {
        v18 += 8;
        goto LABEL_40;
      }
    }
    v74 = v76;
    v75 = 0x800000000LL;
    LODWORD(v72) = v17 + v72;
    a2 = (unsigned int)v72;
    if ( 8 * (unsigned __int64)(unsigned int)v72 > 8 )
      qsort(base, (unsigned int)v72, 8u, (__compar_fn_t)sub_BD8DB0);
  }
  else
  {
LABEL_67:
    base = v73;
    v72 = 0x800000000LL;
    v74 = v76;
    v75 = 0x800000000LL;
  }
  v20 = sub_AA5930(v3);
  v61 = v21;
  if ( v20 == v21 )
  {
LABEL_93:
    if ( v74 != v76 )
      _libc_free(v74, a2);
    if ( base != v73 )
      _libc_free(base, a2);
    v8 = *(_QWORD **)(v3 + 56);
LABEL_16:
    while ( v8 != v6 )
    {
      if ( !v8 )
        BUG();
      if ( v3 != v8[2] )
      {
        v11 = *v2;
        v78 = 1;
        v74 = "Instruction has bogus parent pointer!";
        v77 = 3;
        if ( v11 )
        {
          sub_CA0E80(&v74, v11);
          v12 = *(_BYTE **)(v11 + 32);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
          {
            sub_CB5D20(v11, 10);
          }
          else
          {
            *(_QWORD *)(v11 + 32) = v12 + 1;
            *v12 = 10;
          }
        }
        *((_BYTE *)v2 + 152) = 1;
        return;
      }
      v8 = (_QWORD *)v8[1];
    }
    v9 = *(_BYTE **)(v3 + 72);
    v10 = *(_BYTE *)(v3 + 40);
    if ( v10 == v9[128] )
    {
      if ( v10 )
      {
        if ( sub_AA60B0(v3) )
        {
          v78 = 1;
          v74 = "Basic Block has trailing DbgRecords!";
          v77 = 3;
          sub_BDD6D0(v2, (__int64)&v74);
          if ( *v2 )
            goto LABEL_111;
        }
      }
    }
    else
    {
      v78 = 1;
      v74 = "BB debug format should match parent function";
      v77 = 3;
      sub_BDD6D0(v2, (__int64)&v74);
      if ( *v2 )
      {
        sub_BDBD80((__int64)v2, (_BYTE *)v3);
        v54 = sub_CB59D0(*v2, *(unsigned __int8 *)(v3 + 40));
        v55 = *(_BYTE **)(v54 + 32);
        if ( (unsigned __int64)v55 >= *(_QWORD *)(v54 + 24) )
        {
          sub_CB5D20(v54, 10);
        }
        else
        {
          *(_QWORD *)(v54 + 32) = v55 + 1;
          *v55 = 10;
        }
        sub_BDBD80((__int64)v2, v9);
        v56 = sub_CB59D0(*v2, (unsigned __int8)v9[128]);
        v57 = *(_BYTE **)(v56 + 32);
        if ( (unsigned __int64)v57 >= *(_QWORD *)(v56 + 24) )
        {
          sub_CB5D20(v56, 10);
        }
        else
        {
          *(_QWORD *)(v56 + 32) = v57 + 1;
          *v57 = 10;
        }
      }
    }
    return;
  }
  v59 = v3 + 48;
  v58 = v3;
  v22 = v20;
  while ( 1 )
  {
    if ( (*(_DWORD *)(v22 + 4) & 0x7FFFFFF) != (unsigned __int64)(unsigned int)v72 )
    {
      v49 = v68;
      v70 = 1;
      v68[0] = "PHINode should have one entry for each predecessor of its parent basic block!";
      v69 = 3;
      sub_BDBF70((__int64 *)a1, (__int64)v68);
      if ( *(_QWORD *)a1 )
      {
        v49 = (_BYTE *)v22;
        sub_BDBD80(a1, (_BYTE *)v22);
      }
      goto LABEL_84;
    }
    LODWORD(v75) = 0;
    v23 = HIDWORD(v75);
    v24 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
    if ( v24 > (unsigned __int64)HIDWORD(v75) )
      break;
    if ( v24 )
    {
      v25 = 0;
      goto LABEL_51;
    }
LABEL_88:
    v52 = *(_QWORD *)(v22 + 32);
    if ( !v52 )
      BUG();
    v22 = 0;
    if ( *(_BYTE *)(v52 - 24) == 84 )
      v22 = v52 - 24;
    if ( v61 == v22 )
    {
      v6 = (_QWORD *)v59;
      v2 = (__int64 *)a1;
      v3 = v58;
      goto LABEL_93;
    }
  }
  sub_C8D5F0(&v74, v76, v24, 16);
  v25 = (unsigned int)v75;
  a2 = (unsigned int)v75;
  v32 = (unsigned int)v75;
  v24 = *(_DWORD *)(v22 + 4) & 0x7FFFFFF;
  if ( !v24 )
    goto LABEL_57;
  v23 = HIDWORD(v75);
LABEL_51:
  v26 = 0;
  v27 = 8LL * v24;
  while ( 1 )
  {
    v28 = *(_QWORD *)(v22 - 8);
    v29 = *(_QWORD *)(v28 + 4 * v26);
    v30 = *(_QWORD *)(32LL * *(unsigned int *)(v22 + 72) + v28 + v26);
    if ( v25 + 1 > v23 )
    {
      sub_C8D5F0(&v74, v76, v25 + 1, 16);
      v25 = (unsigned int)v75;
    }
    v31 = (unsigned __int64 *)&v74[16 * v25];
    v26 += 8;
    *v31 = v30;
    v31[1] = v29;
    a2 = (unsigned int)(v75 + 1);
    LODWORD(v75) = v75 + 1;
    if ( v27 == v26 )
      break;
    v23 = HIDWORD(v75);
    v25 = (unsigned int)a2;
  }
  v32 = (unsigned int)a2;
LABEL_57:
  v33 = (unsigned __int64 *)v74;
  v34 = 16 * v32;
  v35 = (unsigned __int64 *)&v74[v34];
  if ( v74 != &v74[v34] )
  {
    v60 = v34;
    _BitScanReverse64(&v36, v34 >> 4);
    sub_BD9170((unsigned __int64 *)v74, (unsigned __int64 *)&v74[v34], 2LL * (int)(63 - (v36 ^ 0x3F)));
    if ( v60 <= 0x100 )
    {
      sub_BD9780(v33, v35);
    }
    else
    {
      sub_BD9780(v33, v33 + 32);
      for ( i = v33 + 32; v35 != i; j[1] = v38 )
      {
        *((_QWORD *)&v38 + 1) = *i;
        *(_QWORD *)&v38 = i[1];
        for ( j = i; ; j[3] = v41 )
        {
          v40 = *(j - 2);
          if ( v38 >= __PAIR128__(v40, *(j - 1)) )
            break;
          *j = v40;
          v41 = *(j - 1);
          j -= 2;
        }
        i += 2;
        *j = *((_QWORD *)&v38 + 1);
      }
    }
    a2 = (unsigned int)v75;
  }
  if ( !(_DWORD)a2 )
    goto LABEL_88;
  v42 = base;
  v43 = 0;
  v44 = *(_QWORD *)v74;
  for ( k = (unsigned __int64 *)(v74 + 16); ; k += 2 )
  {
    if ( *v42 != v44 )
    {
      v47 = a1;
      v49 = v68;
      v62 = v42;
      v66 = k - 2;
      v70 = 1;
      v68[0] = "PHI node entries do not match predecessors!";
      v69 = 3;
      sub_BDBF70((__int64 *)a1, (__int64)v68);
      if ( *(_QWORD *)a1 )
      {
        sub_BDBD80(a1, (_BYTE *)v22);
        v53 = v62;
        if ( *v66 )
        {
          sub_BDBD80(a1, (_BYTE *)*v66);
          v53 = v62;
        }
        v49 = (_BYTE *)*v53;
        if ( *v53 )
          goto LABEL_83;
      }
      goto LABEL_84;
    }
    if ( ++v43 == (_DWORD)a2 )
      goto LABEL_88;
    v46 = *k;
    if ( v44 == *k && k[1] != *(k - 1) )
      break;
    ++v42;
    v44 = v46;
  }
  v47 = a1;
  v48 = k;
  v49 = v68;
  v64 = k - 2;
  v70 = 1;
  v68[0] = "PHI node has multiple entries for the same basic block with different incoming values!";
  v69 = 3;
  sub_BDBF70((__int64 *)a1, (__int64)v68);
  if ( *(_QWORD *)a1 )
  {
    sub_BDBD80(a1, (_BYTE *)v22);
    v50 = v64;
    if ( *v48 )
    {
      sub_BDBD80(a1, (_BYTE *)*v48);
      v50 = v64;
    }
    v51 = (_BYTE *)v48[1];
    if ( v51 )
    {
      v65 = v50;
      sub_BDBD80(a1, v51);
      v50 = v65;
    }
    v49 = (_BYTE *)v50[1];
    if ( v49 )
LABEL_83:
      sub_BDBD80(v47, v49);
  }
LABEL_84:
  if ( v74 != v76 )
    _libc_free(v74, v49);
  if ( base != v73 )
    _libc_free(base, v49);
}
