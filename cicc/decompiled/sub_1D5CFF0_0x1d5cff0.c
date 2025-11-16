// Function: sub_1D5CFF0
// Address: 0x1d5cff0
//
__int64 __fastcall sub_1D5CFF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r15
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  unsigned int v11; // esi
  char v12; // r8
  __int64 v13; // rax
  __int64 v14; // rsi
  char v15; // r8
  _QWORD *v16; // r9
  unsigned __int64 v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rcx
  __int64 result; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rcx
  _BYTE *v24; // rdx
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // rdx
  _BYTE *v30; // r8
  __int64 v31; // r15
  _QWORD *v32; // rax
  _QWORD *v33; // rax
  _BYTE *v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r14
  _BYTE *v38; // rbx
  _BYTE *v39; // r12
  __int64 v40; // r13
  _QWORD *v41; // rax
  _BYTE *v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  char v47; // r9
  unsigned int v48; // r8d
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdi
  char v59; // r8
  unsigned int v60; // esi
  __int64 v61; // r10
  __int64 v62; // rdi
  __int64 v63; // rbx
  __int64 v64; // rdx
  __int64 v65; // [rsp+8h] [rbp-138h]
  __int64 v66; // [rsp+10h] [rbp-130h]
  __int64 v67; // [rsp+18h] [rbp-128h]
  int v68; // [rsp+24h] [rbp-11Ch]
  __int64 v69; // [rsp+28h] [rbp-118h]
  _QWORD *v70; // [rsp+28h] [rbp-118h]
  __int64 v71; // [rsp+30h] [rbp-110h]
  unsigned int v72; // [rsp+30h] [rbp-110h]
  unsigned __int8 v73; // [rsp+38h] [rbp-108h]
  _QWORD v74[4]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v75; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE *v76; // [rsp+68h] [rbp-D8h]
  _BYTE *v77; // [rsp+70h] [rbp-D0h]
  __int64 v78; // [rsp+78h] [rbp-C8h]
  int v79; // [rsp+80h] [rbp-C0h]
  _BYTE v80[184]; // [rsp+88h] [rbp-B8h] BYREF

  v3 = a1;
  v4 = sub_157F280(a1);
  v6 = v5;
  v7 = v4;
  while ( v6 != v7 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    if ( v8 )
    {
      while ( 1 )
      {
        v9 = sub_1648700(v8);
        v10 = v9;
        if ( v9[5] != a2 || *((_BYTE *)v9 + 16) != 77 )
          return 0;
        v11 = *((_DWORD *)v9 + 5) & 0xFFFFFFF;
        if ( v11 )
          break;
LABEL_15:
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          goto LABEL_16;
      }
      v12 = *((_BYTE *)v9 + 23);
      v13 = 3LL * v11;
      v14 = 8LL * v11;
      v15 = v12 & 0x40;
      v16 = &v10[-v13];
      v17 = 0;
      while ( 1 )
      {
        v18 = v16;
        if ( v15 )
          v18 = (_QWORD *)*(v10 - 1);
        v19 = v18[3 * v17 / 8];
        if ( *(_BYTE *)(v19 + 16) > 0x17u
          && v3 == *(_QWORD *)(v19 + 40)
          && v3 != v18[3 * *((unsigned int *)v10 + 14) + 1 + v17 / 8] )
        {
          return 0;
        }
        v17 += 8LL;
        if ( v14 == v17 )
          goto LABEL_15;
      }
    }
LABEL_16:
    v21 = *(_QWORD *)(v7 + 32);
    if ( !v21 )
      BUG();
    v7 = 0;
    if ( *(_BYTE *)(v21 - 8) == 77 )
      v7 = v21 - 24;
  }
  v22 = *(_QWORD *)(a2 + 48);
  if ( !v22 )
    BUG();
  if ( *(_BYTE *)(v22 - 8) != 77 )
    return 1;
  v23 = *(_QWORD *)(v3 + 48);
  v24 = v80;
  v75 = 0;
  v76 = v80;
  v77 = v80;
  v78 = 16;
  v79 = 0;
  if ( !v23 )
    BUG();
  if ( *(_BYTE *)(v23 - 8) != 77 )
  {
    v31 = *(_QWORD *)(v3 + 8);
    if ( v31 )
    {
      while ( 1 )
      {
        v32 = sub_1648700(v31);
        if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
          break;
        v31 = *(_QWORD *)(v31 + 8);
        if ( !v31 )
        {
          v24 = v80;
          v30 = v80;
          goto LABEL_43;
        }
      }
      v33 = sub_1412190((__int64)&v75, v32[5]);
LABEL_39:
      if ( v77 == v76 )
        v34 = &v77[8 * HIDWORD(v78)];
      else
        v34 = &v77[8 * (unsigned int)v78];
      v74[0] = v33;
      v74[1] = v34;
      sub_19E4730((__int64)v74);
      while ( 1 )
      {
        v31 = *(_QWORD *)(v31 + 8);
        if ( !v31 )
          break;
        v35 = sub_1648700(v31);
        if ( (unsigned __int8)(*((_BYTE *)v35 + 16) - 25) <= 9u )
        {
          v33 = sub_1412190((__int64)&v75, v35[5]);
          goto LABEL_39;
        }
      }
      v30 = v77;
      v24 = v76;
    }
    else
    {
      v30 = v80;
    }
    goto LABEL_43;
  }
  if ( (*(_DWORD *)(v23 - 4) & 0xFFFFFFF) != 0 )
  {
    v69 = a2;
    v25 = 0;
    v71 = v23 - 24;
    v26 = v3;
    v27 = 8LL * (*(_DWORD *)(v23 - 4) & 0xFFFFFFF);
    v28 = v23;
    do
    {
      if ( (*(_BYTE *)(v28 - 1) & 0x40) != 0 )
        v29 = *(_QWORD *)(v28 - 32);
      else
        v29 = v71 - 24LL * (*(_DWORD *)(v28 - 4) & 0xFFFFFFF);
      v25 += 8;
      sub_1412190((__int64)&v75, *(_QWORD *)(v25 + v29 + 24LL * *(unsigned int *)(v28 + 32)));
      v30 = v77;
      v24 = v76;
    }
    while ( v27 != v25 );
    v3 = v26;
    a2 = v69;
LABEL_43:
    v68 = *(_DWORD *)(v22 - 4) & 0xFFFFFFF;
    if ( !v68 )
    {
      result = 1;
      goto LABEL_77;
    }
    goto LABEL_44;
  }
  v68 = *(_DWORD *)(v22 - 4) & 0xFFFFFFF;
  if ( !v68 )
    return 1;
  v30 = v80;
LABEL_44:
  v36 = v22 - 24;
  v66 = v22;
  v37 = v3;
  v67 = v36;
  v38 = v30;
  v72 = 0;
  v65 = a2;
  v39 = v24;
  while ( 1 )
  {
    v40 = *(_QWORD *)(sub_13CF970(v67) + 8 * (3LL * *(unsigned int *)(v66 + 32) + v72) + 8);
    if ( v38 == v39 )
      v70 = &v38[8 * HIDWORD(v78)];
    else
      v70 = &v38[8 * (unsigned int)v78];
    v41 = sub_15CC2D0((__int64)&v75, v40);
    v38 = v77;
    v39 = v76;
    if ( v77 == v76 )
      v42 = &v77[8 * HIDWORD(v78)];
    else
      v42 = &v77[8 * (unsigned int)v78];
    for ( ; v42 != (_BYTE *)v41; ++v41 )
    {
      if ( *v41 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    if ( v41 == v70 )
      goto LABEL_75;
    v43 = sub_157F280(v65);
    v45 = v44;
    if ( v43 != v44 )
      break;
LABEL_74:
    v38 = v77;
    v39 = v76;
LABEL_75:
    if ( ++v72 == v68 )
    {
      v30 = v38;
      v24 = v39;
      result = 1;
      goto LABEL_77;
    }
  }
  while ( 1 )
  {
    v46 = 0x17FFFFFFE8LL;
    v47 = *(_BYTE *)(v43 + 23) & 0x40;
    v48 = *(_DWORD *)(v43 + 20) & 0xFFFFFFF;
    if ( v48 )
    {
      v49 = 24LL * *(unsigned int *)(v43 + 56) + 8;
      v50 = 0;
      do
      {
        v51 = v43 - 24LL * v48;
        if ( v47 )
          v51 = *(_QWORD *)(v43 - 8);
        if ( v40 == *(_QWORD *)(v51 + v49) )
        {
          v46 = 24 * v50;
          goto LABEL_61;
        }
        ++v50;
        v49 += 8;
      }
      while ( v48 != (_DWORD)v50 );
      v46 = 0x17FFFFFFE8LL;
    }
LABEL_61:
    if ( v47 )
      v52 = *(_QWORD *)(v43 - 8);
    else
      v52 = v43 - 24LL * v48;
    v53 = *(_QWORD *)(v52 + v46);
    v54 = 0x17FFFFFFE8LL;
    if ( v48 )
    {
      v55 = 0;
      do
      {
        if ( v37 == *(_QWORD *)(v52 + 24LL * *(unsigned int *)(v43 + 56) + 8 * v55 + 8) )
        {
          v54 = 24 * v55;
          goto LABEL_68;
        }
        ++v55;
      }
      while ( v48 != (_DWORD)v55 );
      v54 = 0x17FFFFFFE8LL;
    }
LABEL_68:
    v56 = *(_QWORD *)(v52 + v54);
    if ( *(_BYTE *)(v56 + 16) == 77 && v37 == *(_QWORD *)(v56 + 40) )
    {
      v58 = 0x17FFFFFFE8LL;
      v59 = *(_BYTE *)(v56 + 23) & 0x40;
      v60 = *(_DWORD *)(v56 + 20) & 0xFFFFFFF;
      if ( v60 )
      {
        v61 = 24LL * *(unsigned int *)(v56 + 56) + 8;
        v62 = 0;
        do
        {
          v63 = v56 - 24LL * v60;
          if ( v59 )
            v63 = *(_QWORD *)(v56 - 8);
          if ( v40 == *(_QWORD *)(v63 + v61) )
          {
            v58 = 24 * v62;
            goto LABEL_92;
          }
          ++v62;
          v61 += 8;
        }
        while ( v60 != (_DWORD)v62 );
        v58 = 0x17FFFFFFE8LL;
      }
LABEL_92:
      if ( v59 )
        v64 = *(_QWORD *)(v56 - 8);
      else
        v64 = v56 - 24LL * v60;
      v56 = *(_QWORD *)(v64 + v58);
    }
    if ( v53 != v56 )
      break;
    v57 = *(_QWORD *)(v43 + 32);
    if ( !v57 )
      BUG();
    v43 = 0;
    if ( *(_BYTE *)(v57 - 8) == 77 )
      v43 = v57 - 24;
    if ( v45 == v43 )
      goto LABEL_74;
  }
  v30 = v77;
  v24 = v76;
  result = 0;
LABEL_77:
  if ( v30 != v24 )
  {
    v73 = result;
    _libc_free((unsigned __int64)v30);
    return v73;
  }
  return result;
}
