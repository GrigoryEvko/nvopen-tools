// Function: sub_A84F90
// Address: 0xa84f90
//
char __fastcall sub_A84F90(_QWORD *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  unsigned int v3; // r14d
  __int64 v4; // r12
  __int64 *v5; // rdx
  char v6; // dl
  unsigned __int8 v7; // al
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  unsigned int v10; // r13d
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r12
  size_t *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdi
  size_t v18; // r14
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rdx
  char v21; // al
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  char v24; // r12
  unsigned __int64 v25; // r14
  __int64 *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned int v29; // r14d
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  char *v32; // r14
  size_t v33; // r8
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rcx
  size_t v37; // r8
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  char *v40; // r14
  size_t v41; // r8
  _QWORD *v42; // rax
  _QWORD *v43; // rdi
  __int64 v44; // rbx
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rax
  _BYTE *v47; // rsi
  __int64 v48; // rax
  _QWORD *v49; // rdi
  unsigned __int64 v50; // rax
  _BYTE *v51; // rsi
  __int64 v52; // rax
  _QWORD *v53; // rdi
  char v54; // dl
  char v55; // al
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  char *v58; // r14
  size_t v59; // r8
  _QWORD *v60; // rax
  unsigned __int64 v61; // rax
  _BYTE *v62; // rsi
  __int64 v63; // rax
  _QWORD *v64; // rdi
  unsigned int v67; // [rsp+18h] [rbp-1B8h]
  int v68; // [rsp+1Ch] [rbp-1B4h]
  __int64 v69; // [rsp+20h] [rbp-1B0h]
  __int64 v70; // [rsp+28h] [rbp-1A8h]
  _BYTE *v71; // [rsp+30h] [rbp-1A0h]
  __int64 v72; // [rsp+38h] [rbp-198h]
  size_t n; // [rsp+40h] [rbp-190h]
  __int64 v74; // [rsp+48h] [rbp-188h]
  _QWORD v75[3]; // [rsp+58h] [rbp-178h] BYREF
  char v76; // [rsp+74h] [rbp-15Ch] BYREF
  _BYTE v77[11]; // [rsp+75h] [rbp-15Bh] BYREF
  _QWORD *v78; // [rsp+80h] [rbp-150h] BYREF
  size_t v79; // [rsp+88h] [rbp-148h]
  _QWORD v80[2]; // [rsp+90h] [rbp-140h] BYREF
  _BYTE *v81; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-128h]
  _BYTE v83[64]; // [rsp+B0h] [rbp-120h] BYREF
  _QWORD *v84; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+F8h] [rbp-D8h]
  _QWORD v86[8]; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v87; // [rsp+140h] [rbp-90h] BYREF
  __int64 *v88; // [rsp+148h] [rbp-88h]
  __int64 v89; // [rsp+150h] [rbp-80h]
  int v90; // [rsp+158h] [rbp-78h]
  char v91; // [rsp+15Ch] [rbp-74h]
  char v92; // [rsp+160h] [rbp-70h] BYREF

  v1 = (__int64)"nvvm.annotations";
  v2 = sub_BA8DC0(a1, "nvvm.annotations", 16);
  v69 = v2;
  if ( !v2 )
    return v2;
  v91 = 1;
  v3 = 0;
  v81 = v83;
  v82 = 0x800000000LL;
  v87 = 0;
  v88 = (__int64 *)&v92;
  v89 = 8;
  v90 = 0;
  LODWORD(v2) = sub_B91A00(v2);
  v68 = v2;
  if ( !(_DWORD)v2 )
    goto LABEL_9;
  do
  {
    while ( 1 )
    {
      v1 = v3;
      v4 = sub_B91A10(v69, v3);
      if ( v91 )
      {
        v2 = (__int64)v88;
        v5 = &v88[HIDWORD(v89)];
        if ( v88 != v5 )
        {
          while ( v4 != *(_QWORD *)v2 )
          {
            v2 += 8;
            if ( v5 == (__int64 *)v2 )
              goto LABEL_14;
          }
          goto LABEL_8;
        }
LABEL_14:
        if ( HIDWORD(v89) < (unsigned int)v89 )
          break;
      }
      v1 = v4;
      LOBYTE(v2) = sub_C8CC70(&v87, v4);
      if ( v6 )
        goto LABEL_16;
LABEL_8:
      if ( v68 == ++v3 )
        goto LABEL_9;
    }
    ++HIDWORD(v89);
    *v5 = v4;
    ++v87;
LABEL_16:
    v72 = v4 - 16;
    v7 = *(_BYTE *)(v4 - 16);
    if ( (v7 & 2) != 0 )
      v8 = *(__int64 **)(v4 - 32);
    else
      v8 = (__int64 *)(v72 - 8LL * ((v7 >> 2) & 0xF));
    v2 = *v8;
    if ( !*v8 )
      goto LABEL_8;
    if ( *(_BYTE *)v2 != 1 )
      goto LABEL_8;
    v71 = *(_BYTE **)(v2 + 136);
    if ( *v71 > 3u )
      goto LABEL_8;
    v86[0] = *v8;
    v1 = (__int64)v86;
    v84 = v86;
    v85 = 0x800000001LL;
    LOBYTE(v2) = *(_BYTE *)(v4 - 16);
    v9 = (v2 & 2) != 0 ? *(_DWORD *)(v4 - 24) : (*(_WORD *)(v4 - 16) >> 6) & 0xF;
    if ( v9 <= 1 )
      goto LABEL_8;
    v74 = v4;
    v10 = 2;
    v67 = v3;
    v70 = 16LL * ((v9 - 2) >> 1) + 24;
    v11 = 8;
    if ( (v2 & 2) != 0 )
    {
LABEL_25:
      v12 = *(_QWORD *)(v74 - 32);
      v13 = *(_QWORD *)(v12 + v11);
      goto LABEL_26;
    }
    while ( 1 )
    {
      v12 = v72 - 8LL * (((unsigned __int8)v2 >> 2) & 0xF);
      v13 = *(_QWORD *)(v12 + v11);
LABEL_26:
      v14 = (size_t *)(v12 + 8LL * v10);
      n = *v14;
      v15 = sub_B91420(v13, v1);
      v1 = v15;
      v17 = v16;
      if ( v16 == 6 )
      {
        if ( *(_DWORD *)v15 == 1852990827 && *(_WORD *)(v15 + 4) == 27749 )
        {
          v28 = *(_QWORD *)(n + 136);
          v29 = *(_DWORD *)(v28 + 32);
          if ( v29 <= 0x40 )
            LOBYTE(v2) = *(_QWORD *)(v28 + 24) == 0;
          else
            LOBYTE(v2) = v29 == (unsigned int)sub_C444A0(v28 + 24);
          if ( !(_BYTE)v2 )
          {
            v1 = 1;
            LOBYTE(v2) = sub_CE8040(v71, 1);
          }
          goto LABEL_31;
        }
        goto LABEL_28;
      }
      if ( v16 != 5 )
      {
        switch ( v16 )
        {
          case 0xEuLL:
            if ( *(_QWORD *)v15 != 0x7473756C6378616DLL
              || *(_DWORD *)(v15 + 8) != 1634890341
              || *(_WORD *)(v15 + 12) != 27502 )
            {
              goto LABEL_38;
            }
            goto LABEL_63;
          case 0x12uLL:
            if ( *(_QWORD *)v15 ^ 0x5F72657473756C63LL | *(_QWORD *)(v15 + 8) ^ 0x636F6C625F78616DLL
              || *(_WORD *)(v15 + 16) != 29547 )
            {
              goto LABEL_38;
            }
LABEL_63:
            v30 = *(_QWORD *)(n + 136);
            v31 = *(_QWORD *)(v30 + 24);
            if ( *(_DWORD *)(v30 + 32) > 0x40u )
              v31 = *(_QWORD *)v31;
            if ( !v31 )
            {
              v76 = 48;
              v32 = &v76;
              v78 = v80;
              goto LABEL_67;
            }
            v32 = v77;
            do
            {
              *--v32 = v31 % 0xA + 48;
              v50 = v31;
              v31 /= 0xAu;
            }
            while ( v50 > 9 );
            v51 = (_BYTE *)(v77 - v32);
            v78 = v80;
            v33 = v77 - v32;
            v75[0] = v77 - v32;
            if ( (unsigned __int64)(v77 - v32) > 0xF )
            {
              v52 = sub_22409D0(&v78, v75, 0);
              v33 = v77 - v32;
              v78 = (_QWORD *)v52;
              v53 = (_QWORD *)v52;
              v80[0] = v75[0];
              goto LABEL_96;
            }
            if ( v51 == (_BYTE *)1 )
            {
LABEL_67:
              v33 = 1;
              LOBYTE(v80[0]) = *v32;
              v34 = v80;
            }
            else if ( v51 )
            {
              v53 = v80;
LABEL_96:
              memcpy(v53, v32, v33);
              v33 = v75[0];
              v34 = v78;
            }
            else
            {
              v34 = v80;
            }
            v79 = v33;
            v1 = (__int64)"nvvm.maxclusterrank";
            v35 = 19;
            *((_BYTE *)v34 + v33) = 0;
            v36 = v78;
            v37 = v79;
LABEL_69:
            LOBYTE(v2) = sub_B2CD60(v71, v1, v35, v36, v37);
            if ( v78 != v80 )
            {
              v1 = v80[0] + 1LL;
              LOBYTE(v2) = j_j___libc_free_0(v78, v80[0] + 1LL);
            }
            goto LABEL_31;
          case 8uLL:
            if ( *(_QWORD *)v15 != 0x6D736174636E696DLL )
              goto LABEL_38;
            v38 = *(_QWORD *)(n + 136);
            v39 = *(_QWORD *)(v38 + 24);
            if ( *(_DWORD *)(v38 + 32) > 0x40u )
              v39 = *(_QWORD *)v39;
            if ( !v39 )
            {
              v76 = 48;
              v40 = &v76;
              v78 = v80;
              goto LABEL_77;
            }
            v40 = v77;
            do
            {
              *--v40 = v39 % 0xA + 48;
              v46 = v39;
              v39 /= 0xAu;
            }
            while ( v46 > 9 );
            v47 = (_BYTE *)(v77 - v40);
            v78 = v80;
            v41 = v77 - v40;
            v75[0] = v77 - v40;
            if ( (unsigned __int64)(v77 - v40) > 0xF )
            {
              v48 = sub_22409D0(&v78, v75, 0);
              v41 = v77 - v40;
              v78 = (_QWORD *)v48;
              v49 = (_QWORD *)v48;
              v80[0] = v75[0];
              goto LABEL_91;
            }
            if ( v47 == (_BYTE *)1 )
            {
LABEL_77:
              v41 = 1;
              LOBYTE(v80[0]) = *v40;
              v42 = v80;
            }
            else if ( v47 )
            {
              v49 = v80;
LABEL_91:
              memcpy(v49, v40, v41);
              v41 = v75[0];
              v42 = v78;
            }
            else
            {
              v42 = v80;
            }
            v79 = v41;
            v1 = (__int64)"nvvm.minctasm";
            v35 = 13;
            *((_BYTE *)v42 + v41) = 0;
            v36 = v78;
            v37 = v79;
            goto LABEL_69;
        }
        if ( v16 != 7 )
        {
          if ( v16 > 6 )
          {
LABEL_38:
            if ( *(_DWORD *)v15 == 1853383021 && *(_WORD *)(v15 + 4) == 26996 && *(_BYTE *)(v15 + 6) == 100 )
            {
              v17 = v16 - 7;
              if ( v16 == 8 )
              {
                v54 = *(_BYTE *)(v15 + 7);
                if ( v54 == 120 || v54 == 121 || v54 == 122 )
                {
                  v1 = 12;
                  LOBYTE(v2) = sub_A7C1C0((__int64)"nvvm.maxntid", 12, v54, (__int64)v71, n);
                  goto LABEL_31;
                }
              }
LABEL_141:
              v1 = v15 + 7;
            }
            if ( v17 > 6 )
            {
LABEL_104:
              if ( *(_DWORD *)v1 == 1852925298 && *(_WORD *)(v1 + 4) == 26996 && *(_BYTE *)(v1 + 6) == 100 )
              {
                v17 -= 7LL;
                if ( v17 == 1 )
                {
                  v55 = *(_BYTE *)(v1 + 7);
                  if ( v55 == 120 || v55 == 121 || v55 == 122 )
                  {
                    v1 = 12;
                    LOBYTE(v2) = sub_A7C1C0((__int64)"nvvm.reqntid", 12, v55, (__int64)v71, n);
                    goto LABEL_31;
                  }
                }
                v1 += 7;
              }
            }
          }
          if ( v17 > 0xB && *(_QWORD *)v1 == 0x5F72657473756C63LL && *(_DWORD *)(v1 + 8) == 1601005924 && v17 == 13 )
          {
            v21 = *(_BYTE *)(v1 + 12);
            if ( v21 == 120 || v21 == 121 || v21 == 122 )
            {
              v1 = 16;
              LOBYTE(v2) = sub_A7C1C0((__int64)"nvvm.cluster_dim", 16, v21, (__int64)v71, n);
              goto LABEL_31;
            }
          }
LABEL_28:
          v2 = (unsigned int)v85;
          v18 = *v14;
          v19 = (unsigned int)v85 + 2LL;
          if ( v19 > HIDWORD(v85) )
          {
            v1 = (__int64)v86;
            sub_C8D5F0(&v84, v86, v19, 8);
            v2 = (unsigned int)v85;
          }
          v20 = v84;
          v84[v2] = v13;
          v20[v2 + 1] = v18;
          LODWORD(v85) = v85 + 2;
          goto LABEL_31;
        }
        if ( *(_DWORD *)v15 != 1853383021 || *(_WORD *)(v15 + 4) != 25970 || *(_BYTE *)(v15 + 6) != 103 )
        {
          if ( *(_DWORD *)v15 == 1853383021 && *(_WORD *)(v15 + 4) == 26996 && *(_BYTE *)(v15 + 6) == 100 )
          {
            v17 = 0;
            goto LABEL_141;
          }
          goto LABEL_104;
        }
        v56 = *(_QWORD *)(n + 136);
        v57 = *(_QWORD *)(v56 + 24);
        if ( *(_DWORD *)(v56 + 32) > 0x40u )
          v57 = *(_QWORD *)v57;
        if ( !v57 )
        {
          v76 = 48;
          v58 = &v76;
          v78 = v80;
          goto LABEL_129;
        }
        v58 = v77;
        do
        {
          *--v58 = v57 % 0xA + 48;
          v61 = v57;
          v57 /= 0xAu;
        }
        while ( v61 > 9 );
        v62 = (_BYTE *)(v77 - v58);
        v78 = v80;
        v59 = v77 - v58;
        v75[0] = v77 - v58;
        if ( (unsigned __int64)(v77 - v58) > 0xF )
        {
          v63 = sub_22409D0(&v78, v75, 0);
          v59 = v77 - v58;
          v78 = (_QWORD *)v63;
          v64 = (_QWORD *)v63;
          v80[0] = v75[0];
        }
        else
        {
          if ( v62 == (_BYTE *)1 )
          {
LABEL_129:
            v59 = 1;
            LOBYTE(v80[0]) = *v58;
            v60 = v80;
LABEL_130:
            v79 = v59;
            v1 = (__int64)"nvvm.maxnreg";
            v35 = 12;
            *((_BYTE *)v60 + v59) = 0;
            v36 = v78;
            v37 = v79;
            goto LABEL_69;
          }
          if ( !v62 )
          {
            v60 = v80;
            goto LABEL_130;
          }
          v64 = v80;
        }
        memcpy(v64, v58, v59);
        v59 = v75[0];
        v60 = v78;
        goto LABEL_130;
      }
      if ( *(_DWORD *)v15 != 1734962273 || *(_BYTE *)(v15 + 4) != 110 )
        goto LABEL_28;
      v22 = *(_QWORD *)(n + 136);
      v23 = *(_QWORD *)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
        v23 = *(_QWORD *)v23;
      v24 = -1;
      v25 = v23 >> 16;
      if ( (_WORD)v23 )
      {
        _BitScanReverse64(&v23, (unsigned __int16)v23);
        v1 = 63;
        v24 = 63 - (v23 ^ 0x3F);
      }
      v26 = (__int64 *)sub_BD5C60(v71, v1, v22);
      v27 = sub_A77A60(v26, v24);
      v1 = (unsigned int)v25;
      LOBYTE(v2) = sub_B2CCF0(v71, (unsigned int)v25, v27);
LABEL_31:
      v10 += 2;
      v11 += 16;
      if ( v70 == v11 )
        break;
      LOBYTE(v2) = *(_BYTE *)(v74 - 16);
      if ( (v2 & 2) != 0 )
        goto LABEL_25;
    }
    v3 = v67;
    v43 = v84;
    if ( (unsigned int)v85 > 1uLL )
    {
      v1 = (__int64)v84;
      v44 = sub_B9C770(*a1, v84, (unsigned int)v85, 0, 1);
      v2 = (unsigned int)v82;
      v45 = (unsigned int)v82 + 1LL;
      if ( v45 > HIDWORD(v82) )
      {
        v1 = (__int64)v83;
        sub_C8D5F0(&v81, v83, v45, 8);
        v2 = (unsigned int)v82;
      }
      *(_QWORD *)&v81[8 * v2] = v44;
      v43 = v84;
      LODWORD(v82) = v82 + 1;
    }
    if ( v43 == v86 )
      goto LABEL_8;
    LOBYTE(v2) = _libc_free(v43, v1);
    v3 = v67 + 1;
  }
  while ( v68 != v67 + 1 );
LABEL_9:
  if ( !v91 )
    LOBYTE(v2) = _libc_free(v88, v1);
  if ( v81 != v83 )
    LOBYTE(v2) = _libc_free(v81, v1);
  return v2;
}
