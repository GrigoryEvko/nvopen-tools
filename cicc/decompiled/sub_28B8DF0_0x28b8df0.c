// Function: sub_28B8DF0
// Address: 0x28b8df0
//
__int64 __fastcall sub_28B8DF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // rsi
  _QWORD *v7; // r8
  __int64 v8; // rdi
  char *v9; // rsi
  char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  char *v13; // rdi
  _BYTE *v14; // rdx
  __int64 v15; // r11
  _QWORD *v16; // r12
  int v17; // r13d
  unsigned int v18; // ecx
  _QWORD *v19; // r10
  _BYTE *v20; // r14
  int v22; // r10d
  _BYTE *v23; // rdx
  __int64 v24; // r11
  _QWORD *v25; // r12
  int v26; // r13d
  unsigned int v27; // ecx
  _QWORD *v28; // r10
  _BYTE *v29; // r14
  int v30; // r10d
  _BYTE *v31; // rcx
  __int64 v32; // r11
  _QWORD *v33; // r12
  int v34; // r13d
  unsigned int v35; // edx
  _QWORD *v36; // r10
  _BYTE *v37; // r14
  int v38; // r10d
  _BYTE *v39; // rcx
  __int64 v40; // r11
  _QWORD *v41; // r12
  int v42; // r13d
  unsigned int v43; // edx
  _QWORD *v44; // r10
  _BYTE *v45; // r14
  int v46; // r10d
  __int64 v47; // r10
  __int64 v48; // r10
  __int64 v49; // r10
  __int64 v50; // r10
  _BYTE *v51; // rcx
  __int64 v52; // r10
  _QWORD *v53; // r11
  int v54; // edi
  unsigned int v55; // edx
  _BYTE *v56; // r9
  _BYTE *v57; // rcx
  _BYTE *v58; // rcx
  __int64 v59; // r10
  __int64 v60; // rdi
  _QWORD *v61; // rbx
  unsigned int v62; // edx
  _QWORD *v63; // rdi
  _BYTE *v64; // r9
  __int64 v65; // r9
  _QWORD *v66; // r11
  unsigned int v67; // edx
  _QWORD *v68; // rdi
  _BYTE *v69; // r10
  int v70; // r15d
  int v71; // r15d
  int v72; // r15d
  int v73; // r15d
  __int64 v74; // rdi
  __int64 v75; // rdi
  int v76; // edi
  int v77; // r11d
  int v78; // edi
  int v79; // r12d
  int v80; // r8d
  int v81; // r12d
  __int64 v82; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v83[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = a2;
  v82 = a2;
  if ( (unsigned __int8)sub_B46490(a2) )
  {
    v6 = *(_QWORD *)(a1 + 104);
    v83[1] = a3;
    v83[0] = &v82;
    if ( sub_28B4B30((__int64)v83, v6) || sub_28B4B30((__int64)v83, *(_QWORD *)(a1 + 144)) )
    {
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    v4 = v82;
  }
  v8 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
  {
    v10 = *(char **)(v4 - 8);
    v9 = &v10[v8];
  }
  else
  {
    v9 = (char *)v4;
    v10 = (char *)(v4 - v8);
  }
  v11 = v8 >> 5;
  v12 = v8 >> 7;
  if ( v12 )
  {
    LODWORD(v7) = a1 + 88;
    v13 = &v10[128 * v12];
    while ( 1 )
    {
      v14 = *(_BYTE **)v10;
      if ( **(_BYTE **)v10 <= 0x1Cu )
        goto LABEL_20;
      if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
      {
        v15 = a1 + 24;
        v16 = (_QWORD *)(a1 + 88);
        v17 = 7;
      }
      else
      {
        v15 = *(_QWORD *)(a1 + 24);
        v47 = *(unsigned int *)(a1 + 32);
        v16 = (_QWORD *)(v15 + 8 * v47);
        if ( !(_DWORD)v47 )
          goto LABEL_20;
        v17 = v47 - 1;
      }
      v18 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v19 = (_QWORD *)(v15 + 8LL * v18);
      v20 = (_BYTE *)*v19;
      if ( v14 == (_BYTE *)*v19 )
      {
LABEL_14:
        if ( v19 != v16 )
          goto LABEL_15;
      }
      else
      {
        v22 = 1;
        while ( v20 != (_BYTE *)-4096LL )
        {
          v70 = v22 + 1;
          v18 = v17 & (v22 + v18);
          v19 = (_QWORD *)(v15 + 8LL * v18);
          v20 = (_BYTE *)*v19;
          if ( v14 == (_BYTE *)*v19 )
            goto LABEL_14;
          v22 = v70;
        }
      }
LABEL_20:
      v23 = (_BYTE *)*((_QWORD *)v10 + 4);
      if ( *v23 <= 0x1Cu )
        goto LABEL_28;
      if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
      {
        v24 = a1 + 24;
        v25 = (_QWORD *)(a1 + 88);
        v26 = 7;
      }
      else
      {
        v24 = *(_QWORD *)(a1 + 24);
        v48 = *(unsigned int *)(a1 + 32);
        v25 = (_QWORD *)(v24 + 8 * v48);
        if ( !(_DWORD)v48 )
          goto LABEL_28;
        v26 = v48 - 1;
      }
      v27 = v26 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v28 = (_QWORD *)(v24 + 8LL * v27);
      v29 = (_BYTE *)*v28;
      if ( v23 == (_BYTE *)*v28 )
      {
LABEL_24:
        if ( v28 != v25 )
        {
          LOBYTE(v7) = v9 == v10 + 32;
          return (unsigned int)v7;
        }
      }
      else
      {
        v30 = 1;
        while ( v29 != (_BYTE *)-4096LL )
        {
          v71 = v30 + 1;
          v27 = v26 & (v30 + v27);
          v28 = (_QWORD *)(v24 + 8LL * v27);
          v29 = (_BYTE *)*v28;
          if ( v23 == (_BYTE *)*v28 )
            goto LABEL_24;
          v30 = v71;
        }
      }
LABEL_28:
      v31 = (_BYTE *)*((_QWORD *)v10 + 8);
      if ( *v31 <= 0x1Cu )
        goto LABEL_36;
      if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
      {
        v32 = a1 + 24;
        v33 = (_QWORD *)(a1 + 88);
        v34 = 7;
      }
      else
      {
        v32 = *(_QWORD *)(a1 + 24);
        v49 = *(unsigned int *)(a1 + 32);
        v33 = (_QWORD *)(v32 + 8 * v49);
        if ( !(_DWORD)v49 )
          goto LABEL_36;
        v34 = v49 - 1;
      }
      v35 = v34 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v36 = (_QWORD *)(v32 + 8LL * v35);
      v37 = (_BYTE *)*v36;
      if ( v31 == (_BYTE *)*v36 )
      {
LABEL_32:
        if ( v36 != v33 )
        {
          LOBYTE(v7) = v9 == v10 + 64;
          return (unsigned int)v7;
        }
      }
      else
      {
        v38 = 1;
        while ( v37 != (_BYTE *)-4096LL )
        {
          v72 = v38 + 1;
          v35 = v34 & (v38 + v35);
          v36 = (_QWORD *)(v32 + 8LL * v35);
          v37 = (_BYTE *)*v36;
          if ( v31 == (_BYTE *)*v36 )
            goto LABEL_32;
          v38 = v72;
        }
      }
LABEL_36:
      v39 = (_BYTE *)*((_QWORD *)v10 + 12);
      if ( *v39 > 0x1Cu )
      {
        if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
        {
          v40 = a1 + 24;
          v41 = (_QWORD *)(a1 + 88);
          v42 = 7;
          goto LABEL_39;
        }
        v40 = *(_QWORD *)(a1 + 24);
        v50 = *(unsigned int *)(a1 + 32);
        v41 = (_QWORD *)(v40 + 8 * v50);
        if ( (_DWORD)v50 )
        {
          v42 = v50 - 1;
LABEL_39:
          v43 = v42 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v44 = (_QWORD *)(v40 + 8LL * v43);
          v45 = (_BYTE *)*v44;
          if ( v39 == (_BYTE *)*v44 )
          {
LABEL_40:
            if ( v44 != v41 )
            {
              LOBYTE(v7) = v9 == v10 + 96;
              return (unsigned int)v7;
            }
          }
          else
          {
            v46 = 1;
            while ( v45 != (_BYTE *)-4096LL )
            {
              v73 = v46 + 1;
              v43 = v42 & (v46 + v43);
              v44 = (_QWORD *)(v40 + 8LL * v43);
              v45 = (_BYTE *)*v44;
              if ( v39 == (_BYTE *)*v44 )
                goto LABEL_40;
              v46 = v73;
            }
          }
        }
      }
      v10 += 128;
      if ( v10 == v13 )
      {
        v11 = (v9 - v10) >> 5;
        break;
      }
    }
  }
  if ( v11 != 2 )
  {
    if ( v11 != 3 )
    {
      if ( v11 != 1 )
      {
LABEL_49:
        LODWORD(v7) = 1;
        return (unsigned int)v7;
      }
      goto LABEL_66;
    }
    v51 = *(_BYTE **)v10;
    if ( **(_BYTE **)v10 <= 0x1Cu )
      goto LABEL_63;
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
      v52 = a1 + 24;
      v53 = (_QWORD *)(a1 + 88);
      v54 = 7;
    }
    else
    {
      v52 = *(_QWORD *)(a1 + 24);
      v75 = *(unsigned int *)(a1 + 32);
      v53 = (_QWORD *)(v52 + 8 * v75);
      if ( !(_DWORD)v75 )
      {
LABEL_63:
        v10 += 32;
        goto LABEL_64;
      }
      v54 = v75 - 1;
    }
    v55 = v54 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v7 = (_QWORD *)(v52 + 8LL * v55);
    v56 = (_BYTE *)*v7;
    if ( v51 == (_BYTE *)*v7 )
    {
LABEL_62:
      if ( v7 != v53 )
      {
LABEL_15:
        LOBYTE(v7) = v9 == v10;
        return (unsigned int)v7;
      }
    }
    else
    {
      v80 = 1;
      while ( v56 != (_BYTE *)-4096LL )
      {
        v81 = v80 + 1;
        v55 = v54 & (v80 + v55);
        v7 = (_QWORD *)(v52 + 8LL * v55);
        v56 = (_BYTE *)*v7;
        if ( v51 == (_BYTE *)*v7 )
          goto LABEL_62;
        v80 = v81;
      }
    }
    goto LABEL_63;
  }
LABEL_64:
  v57 = *(_BYTE **)v10;
  if ( **(_BYTE **)v10 > 0x1Cu )
  {
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
      v65 = a1 + 24;
      v66 = (_QWORD *)(a1 + 88);
      LODWORD(v7) = 7;
    }
    else
    {
      v65 = *(_QWORD *)(a1 + 24);
      v74 = *(unsigned int *)(a1 + 32);
      v66 = (_QWORD *)(v65 + 8 * v74);
      if ( !(_DWORD)v74 )
        goto LABEL_65;
      LODWORD(v7) = v74 - 1;
    }
    v67 = (unsigned int)v7 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
    v68 = (_QWORD *)(v65 + 8LL * v67);
    v69 = (_BYTE *)*v68;
    if ( v57 == (_BYTE *)*v68 )
    {
LABEL_77:
      if ( v68 != v66 )
      {
        LOBYTE(v7) = v10 == v9;
        return (unsigned int)v7;
      }
    }
    else
    {
      v78 = 1;
      while ( v69 != (_BYTE *)-4096LL )
      {
        v79 = v78 + 1;
        v67 = (unsigned int)v7 & (v78 + v67);
        v68 = (_QWORD *)(v65 + 8LL * v67);
        v69 = (_BYTE *)*v68;
        if ( v57 == (_BYTE *)*v68 )
          goto LABEL_77;
        v78 = v79;
      }
    }
  }
LABEL_65:
  v10 += 32;
LABEL_66:
  v58 = *(_BYTE **)v10;
  LODWORD(v7) = 1;
  if ( **(_BYTE **)v10 <= 0x1Cu )
    return (unsigned int)v7;
  if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
  {
    v59 = a1 + 24;
    LODWORD(v7) = 7;
    v61 = (_QWORD *)(a1 + 88);
  }
  else
  {
    v59 = *(_QWORD *)(a1 + 24);
    v60 = *(unsigned int *)(a1 + 32);
    v61 = (_QWORD *)(v59 + 8 * v60);
    if ( !(_DWORD)v60 )
      goto LABEL_49;
    LODWORD(v7) = v60 - 1;
  }
  v62 = (unsigned int)v7 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
  v63 = (_QWORD *)(v59 + 8LL * v62);
  v64 = (_BYTE *)*v63;
  if ( v58 != (_BYTE *)*v63 )
  {
    v76 = 1;
    while ( v64 != (_BYTE *)-4096LL )
    {
      v77 = v76 + 1;
      v62 = (unsigned int)v7 & (v76 + v62);
      v63 = (_QWORD *)(v59 + 8LL * v62);
      v64 = (_BYTE *)*v63;
      if ( v58 == (_BYTE *)*v63 )
        goto LABEL_71;
      v76 = v77;
    }
    goto LABEL_49;
  }
LABEL_71:
  LOBYTE(v7) = v10 == v9;
  if ( v63 == v61 )
    LODWORD(v7) = 1;
  return (unsigned int)v7;
}
