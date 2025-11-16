// Function: sub_1438F00
// Address: 0x1438f00
//
__int64 __fastcall sub_1438F00(__int64 a1)
{
  unsigned int v1; // r13d
  _QWORD *v2; // rbx
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  char v9; // bl
  char v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // rdx
  const void *v13; // r15
  __int64 v14; // r12
  int v15; // eax
  __int64 v16; // r13
  unsigned __int8 v17; // bl
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // r14
  __int64 v25; // r12
  char v26; // cl
  bool v27; // al
  char v28; // al
  bool v29; // si
  char v30; // dl
  int v31; // eax
  char v32; // dl
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // eax
  int v36; // eax

  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    sub_15E08E0(a1);
    v2 = *(_QWORD **)(a1 + 88);
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      sub_15E08E0(a1);
    v3 = *(_QWORD **)(a1 + 88);
  }
  else
  {
    v2 = *(_QWORD **)(a1 + 88);
    v3 = v2;
  }
  v4 = &v3[5 * *(_QWORD *)(a1 + 96)];
  if ( v4 == v2 )
  {
    v20 = sub_1649960(a1);
    if ( v21 == 24 )
    {
      if ( !(*(_QWORD *)v20 ^ 0x7475615F636A626FLL | *(_QWORD *)(v20 + 8) ^ 0x657361656C65726FLL) )
      {
        v1 = 7;
        if ( *(_QWORD *)(v20 + 16) == 0x687375506C6F6F50LL )
          return v1;
      }
    }
    else if ( v21 == 13 && *(_QWORD *)v20 == 0x72612E676E616C63LL && *(_DWORD *)(v20 + 8) == 1937059427 )
    {
      v1 = 20;
      if ( *(_BYTE *)(v20 + 12) == 101 )
        return v1;
    }
    return 21;
  }
  if ( v4 == v2 + 5 )
  {
    if ( *(_BYTE *)(*v2 + 8LL) != 15 )
      return 21;
    v16 = *(_QWORD *)(*v2 + 24LL);
    v17 = sub_1642F90(v16, 8);
    if ( !v17 )
    {
      if ( *(_BYTE *)(v16 + 8) != 15 || !(unsigned __int8)sub_1642F90(*(_QWORD *)(v16 + 24), 8) )
        return 21;
      v18 = sub_1649960(a1);
      if ( v19 == 21 )
      {
        if ( !(*(_QWORD *)v18 ^ 0x616F6C5F636A626FLL | *(_QWORD *)(v18 + 8) ^ 0x7465526B61655764LL)
          && *(_DWORD *)(v18 + 16) == 1701734753 )
        {
          v1 = 12;
          if ( *(_BYTE *)(v18 + 20) == 100 )
            return v1;
        }
        return 21;
      }
      if ( v19 == 13 )
      {
        if ( *(_QWORD *)v18 == 0x616F6C5F636A626FLL && *(_DWORD *)(v18 + 8) == 1634031460 )
        {
          v1 = 15;
          if ( *(_BYTE *)(v18 + 12) == 107 )
            return v1;
        }
        return 21;
      }
      v1 = 21;
      if ( v19 == 16 )
        return (*(_QWORD *)(v18 + 8) ^ 0x6B616557796F7274LL | *(_QWORD *)v18 ^ 0x7365645F636A626FLL) == 0 ? 18 : 21;
      return v1;
    }
    v22 = (_QWORD *)sub_1649960(a1);
    v24 = v22;
    v25 = v23;
    switch ( v23 )
    {
      case 11LL:
        if ( *v22 == 0x7465725F636A626FLL && *((_WORD *)v22 + 4) == 26977 && *((_BYTE *)v22 + 10) == 110 )
        {
          v26 = v17;
          v17 = 0;
          goto LABEL_62;
        }
        break;
      case 34LL:
        if ( *v22 ^ 0x7465725F636A626FLL | v22[1] ^ 0x726F7475416E6961LL
          || v22[2] ^ 0x5264657361656C65LL | v22[3] ^ 0x6C61566E72757465LL
          || *((_WORD *)v22 + 16) != 25973 )
        {
          v26 = 0;
          v1 = 1;
        }
        else
        {
          v26 = v17;
          v1 = 1;
          v17 = 0;
        }
        goto LABEL_63;
      case 39LL:
        v31 = memcmp(v22, "objc_unsafeClaimAutoreleasedReturnValue", 0x27u);
        if ( !v31 )
        {
          v26 = v17;
          v1 = 2;
          v17 = 0;
          goto LABEL_65;
        }
        v30 = 0;
        v26 = 0;
        v1 = 0;
LABEL_64:
        LOBYTE(v31) = v25 == 27;
        if ( v30 && *v24 == 0x6C65725F636A626FLL && *((_DWORD *)v24 + 2) == 1702060389 )
        {
          v26 = v30;
          v1 = 4;
          v27 = v25 == 23;
          v17 = 0;
          goto LABEL_50;
        }
LABEL_65:
        v32 = v17 & v31;
        v27 = v25 == 23;
        if ( v32 )
        {
          if ( !(*v24 ^ 0x7475615F636A626FLL | v24[1] ^ 0x657361656C65726FLL)
            && v24[2] == 0x61566E7275746552LL
            && *((_WORD *)v24 + 12) == 30060
            && *((_BYTE *)v24 + 26) == 101 )
          {
            v26 = v32;
            v17 = 0;
            v1 = 6;
            goto LABEL_52;
          }
          goto LABEL_51;
        }
        goto LABEL_50;
      case 16LL:
        if ( *v22 ^ 0x7465725F636A626FLL | v22[1] ^ 0x6B636F6C426E6961LL )
        {
          if ( *v22 ^ 0x7475615F636A626FLL | v22[1] ^ 0x657361656C65726FLL )
          {
            v27 = 0;
            v26 = 0;
            v1 = 0;
          }
          else
          {
            v26 = v17;
            v27 = 0;
            v1 = 5;
            v17 = 0;
          }
          goto LABEL_51;
        }
        v26 = v17;
        v27 = 0;
        v17 = 0;
        v1 = 3;
LABEL_50:
        if ( (v17 & v27) != 0 )
        {
          if ( *v24 ^ 0x7475615F636A626FLL | v24[1] ^ 0x657361656C65726FLL
            || *((_DWORD *)v24 + 4) != 1819242320
            || *((_WORD *)v24 + 10) != 28496
            || *((_BYTE *)v24 + 22) != 112 )
          {
            goto LABEL_52;
          }
          v26 = v17 & v27;
          v17 = 0;
          v1 = 8;
          goto LABEL_53;
        }
LABEL_51:
        if ( (v17 & (v25 == 19)) == 0 )
        {
LABEL_52:
          if ( (v17 & (v25 == 21)) == 0 )
            goto LABEL_53;
          if ( !(*v24 ^ 0x726E755F636A626FLL | v24[1] ^ 0x4F64656E69617465LL)
            && *((_DWORD *)v24 + 4) == 1667590754
            && *((_BYTE *)v24 + 20) == 116 )
          {
            v26 = v17 & (v25 == 21);
            v1 = 9;
            v29 = v25 == 33;
            v17 = 0;
            goto LABEL_55;
          }
LABEL_54:
          v28 = v17 & v27;
          v29 = v25 == 33;
          if ( v28
            && !(*v24 ^ 0x7465725F636A626FLL | v24[1] ^ 0x6F7475615F6E6961LL)
            && *((_DWORD *)v24 + 4) == 1701602674
            && *((_WORD *)v24 + 10) == 29537
            && *((_BYTE *)v24 + 22) == 101 )
          {
            v26 = v28;
            v17 = 0;
            v1 = 10;
            goto LABEL_56;
          }
LABEL_55:
          if ( (v17 & v29) != 0 )
          {
            if ( !(*v24 ^ 0x7465725F636A626FLL | v24[1] ^ 0x726F7475416E6961LL)
              && !(v24[2] ^ 0x6552657361656C65LL | v24[3] ^ 0x756C61566E727574LL)
              && *((_BYTE *)v24 + 32) == 101 )
            {
              return 11;
            }
            goto LABEL_57;
          }
          goto LABEL_56;
        }
        if ( !(*v24 ^ 0x7465725F636A626FLL | v24[1] ^ 0x6A624F64656E6961LL)
          && *((_WORD *)v24 + 8) == 25445
          && *((_BYTE *)v24 + 18) == 116 )
        {
          v26 = v17 & (v25 == 19);
          v17 = 0;
          v1 = 9;
          goto LABEL_54;
        }
LABEL_53:
        if ( (v17 & (v25 == 22)) != 0 )
        {
          if ( !(*v24 ^ 0x726E755F636A626FLL | v24[1] ^ 0x5064656E69617465LL)
            && *((_DWORD *)v24 + 4) == 1953393007
            && *((_WORD *)v24 + 10) == 29285 )
          {
            v26 = v17 & (v25 == 22);
            v17 = 0;
            v1 = 9;
          }
          else if ( !(*v24 ^ 0x7465725F636A626FLL | v24[1] ^ 0x726F7475416E6961LL)
                 && *((_DWORD *)v24 + 4) == 1634036837
                 && *((_WORD *)v24 + 10) == 25971 )
          {
            v26 = v17 & (v25 == 22);
            v17 = 0;
            v1 = 10;
            goto LABEL_57;
          }
LABEL_56:
          if ( (v17 & (v25 == 15)) != 0 )
          {
            if ( *v24 != 0x6E79735F636A626FLL
              || *((_DWORD *)v24 + 2) != 1852137315
              || *((_WORD *)v24 + 6) != 25972
              || *((_BYTE *)v24 + 14) != 114 )
            {
LABEL_58:
              if ( !v26 )
                return 21;
              return v1;
            }
            return 23;
          }
LABEL_57:
          if ( ((v25 == 14) & v17) == 0 )
            goto LABEL_58;
          if ( *v24 != 0x6E79735F636A626FLL || *((_DWORD *)v24 + 2) != 2019909475 || *((_WORD *)v24 + 6) != 29801 )
            return 21;
          return 23;
        }
        goto LABEL_54;
    }
    v26 = 0;
LABEL_62:
    v1 = 0;
LABEL_63:
    v30 = v17 & (v23 == 12);
    goto LABEL_64;
  }
  if ( v4 != v2 + 10 )
    return 21;
  if ( *(_BYTE *)(*v2 + 8LL) != 15 )
    return 21;
  v6 = *(_QWORD *)(*v2 + 24LL);
  if ( *(_BYTE *)(v6 + 8) != 15 )
    return 21;
  if ( !(unsigned __int8)sub_1642F90(*(_QWORD *)(v6 + 24), 8) )
    return 21;
  v7 = v2[5];
  if ( *(_BYTE *)(v7 + 8) != 15 )
    return 21;
  v8 = *(_QWORD *)(v7 + 24);
  v9 = sub_1642F90(v8, 8);
  if ( !v9 )
  {
    if ( *(_BYTE *)(v8 + 8) == 15 )
    {
      v10 = sub_1642F90(*(_QWORD *)(v8 + 24), 8);
      if ( v10 )
      {
        v11 = (_QWORD *)sub_1649960(a1);
        v13 = v11;
        v14 = v12;
        switch ( v12 )
        {
          case 13LL:
            if ( *v11 == 0x766F6D5F636A626FLL && *((_DWORD *)v11 + 2) == 1634031461 && *((_BYTE *)v11 + 12) == 107 )
            {
              v9 = 1;
              v1 = 16;
            }
            else
            {
              v1 = 0;
              v35 = memcmp(v11, "objc_copyWeak", 0xDu);
              v9 = v35 == 0;
              if ( !v35 )
                v1 = 17;
            }
            v10 = v9 ^ 1;
            break;
          case 35LL:
            v1 = 24;
            v36 = memcmp(v11, "llvm.arc.annotation.topdown.bbstart", 0x23u);
            v9 = v36 == 0;
            v10 = v36 != 0;
            break;
          case 33LL:
            v1 = 24;
            v9 = memcmp(v11, "llvm.arc.annotation.topdown.bbend", 0x21u) == 0;
LABEL_21:
            if ( !v9 )
              return 21;
            return v1;
          case 36LL:
            v1 = 0;
            v15 = memcmp(v11, "llvm.arc.annotation.bottomup.bbstart", 0x24u);
            v9 = v15 == 0;
            if ( !v15 )
              v1 = 24;
            goto LABEL_21;
        }
        if ( v14 == 34 && v10 )
          return memcmp(v13, "llvm.arc.annotation.bottomup.bbend", 0x22u) == 0 ? 24 : 21;
        goto LABEL_21;
      }
    }
    return 21;
  }
  v33 = sub_1649960(a1);
  if ( v34 == 14 )
  {
    if ( *(_QWORD *)v33 == 0x6F74735F636A626FLL && *(_DWORD *)(v33 + 8) == 1700226418 )
    {
      v1 = 13;
      if ( *(_WORD *)(v33 + 12) == 27489 )
        return v1;
    }
    return 21;
  }
  if ( v34 == 13 )
  {
    if ( *(_QWORD *)v33 == 0x696E695F636A626FLL && *(_DWORD *)(v33 + 8) == 1634031476 )
    {
      v1 = 14;
      if ( *(_BYTE *)(v33 + 12) == 107 )
        return v1;
    }
    return 21;
  }
  v1 = 21;
  if ( v34 == 16 )
    return (*(_QWORD *)(v33 + 8) ^ 0x676E6F7274536572LL | *(_QWORD *)v33 ^ 0x6F74735F636A626FLL) == 0 ? 19 : 21;
  return v1;
}
