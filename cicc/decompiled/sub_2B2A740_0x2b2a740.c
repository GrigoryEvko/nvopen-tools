// Function: sub_2B2A740
// Address: 0x2b2a740
//
__int64 __fastcall sub_2B2A740(__int64 **a1)
{
  unsigned __int64 v1; // rax
  __int64 *v2; // rcx
  __int64 v3; // r8
  char *v4; // rdx
  char v5; // dl
  unsigned int v6; // esi
  _QWORD *v7; // r10
  int v8; // r9d
  __int64 *v9; // rbx
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  unsigned __int8 *v12; // rdx
  unsigned __int8 **v13; // rdx
  __int64 v14; // rdi
  _BYTE **v15; // r11
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r12
  unsigned __int8 **v19; // rax
  unsigned __int8 v20; // cl
  unsigned __int8 **v21; // rax
  __int64 v22; // rcx
  unsigned __int8 v23; // si
  unsigned __int8 **v24; // rdi
  unsigned __int8 *v25; // rcx
  unsigned __int8 **v26; // rsi
  unsigned __int8 v27; // al
  unsigned __int8 *v28; // rcx
  unsigned __int8 v29; // al
  unsigned __int8 *v30; // rcx
  unsigned __int8 v31; // al
  unsigned __int8 v32; // al
  unsigned __int8 **v34; // r14
  unsigned __int8 v35; // si
  unsigned __int8 v36; // si
  unsigned __int8 v37; // si
  unsigned __int8 v38; // cl
  unsigned __int8 v39; // cl
  unsigned __int8 v40; // cl
  unsigned __int8 v41; // cl
  __int64 v42; // rcx
  unsigned __int8 v43; // cl
  unsigned __int8 v44; // al
  unsigned __int8 v45; // cl
  unsigned __int8 v46; // cl
  unsigned __int8 v47; // al
  unsigned __int8 v48; // al
  unsigned __int8 v49; // cl
  unsigned __int8 v50; // cl

  v1 = *((unsigned int *)a1 + 2);
  if ( v1 <= 0xA )
    return 1;
  v2 = *a1;
  v3 = **a1;
  v4 = *(char **)(v3 + 416);
  if ( !v4 || !*(_QWORD *)(v3 + 424) || *(_DWORD *)(v3 + 104) == 3 )
    return 1;
  v5 = *v4;
  if ( v5 == 84 || v5 == 62 )
  {
    if ( !*(_DWORD *)(v3 + 152) )
    {
      if ( v5 == 84 && *(_DWORD *)(v3 + 8) == 2 && *(_DWORD *)(v3 + 248) > 0xCu )
        return 0;
      goto LABEL_13;
    }
    return 1;
  }
  v6 = *(_DWORD *)(v3 + 120);
  if ( !v6 )
    v6 = *(_DWORD *)(v3 + 8);
  if ( v6 > 2 || v5 != 76 && v5 != 82 || *(_DWORD *)(v3 + 152) )
    return 1;
LABEL_13:
  v7 = v2 + 1;
  v8 = 0;
  v9 = &v2[v1];
  while ( 1 )
  {
    v11 = *v7;
    v12 = *(unsigned __int8 **)(*v7 + 416LL);
    if ( v12 && *(_QWORD *)(v11 + 424) )
    {
      v10 = *v12;
      if ( v10 == 61 )
      {
        if ( *(_DWORD *)(v11 + 152) || *(_DWORD *)(v11 + 104) == 3 )
          return 1;
      }
      else if ( v10 != 63 && (unsigned int)v10 - 42 > 0x11 )
      {
        if ( v10 != 84 )
          return 1;
        v8 = 1;
        if ( *(_DWORD *)(v3 + 8) == 2 && *(_DWORD *)(v11 + 248) > 0xCu )
          return 0;
      }
      goto LABEL_20;
    }
    v13 = *(unsigned __int8 ***)v11;
    v14 = 8LL * *(unsigned int *)(v11 + 8);
    v15 = (_BYTE **)(*(_QWORD *)v11 + v14);
    v16 = v14 >> 3;
    v17 = v14 >> 5;
    v18 = v16;
    if ( v17 )
    {
      v19 = *(unsigned __int8 ***)v11;
      while ( 1 )
      {
        v20 = **v19;
        if ( v20 > 0x15u && v20 != 84 )
          break;
        v39 = *v19[1];
        if ( v39 > 0x15u && v39 != 84 )
        {
          ++v19;
          break;
        }
        v40 = *v19[2];
        if ( v40 > 0x15u && v40 != 84 )
        {
          v19 += 2;
          break;
        }
        v41 = *v19[3];
        if ( v41 > 0x15u && v41 != 84 )
        {
          v19 += 3;
          break;
        }
        v19 += 4;
        if ( &v13[4 * v17] == v19 )
        {
          v42 = v15 - v19;
          goto LABEL_102;
        }
      }
      if ( v15 == v19 )
        goto LABEL_20;
      v21 = v13;
      v22 = v17;
LABEL_28:
      while ( 1 )
      {
        v23 = **v21;
        if ( v23 <= 0x1Cu || (unsigned int)v23 - 42 > 0x11 && v23 != 84 )
          goto LABEL_29;
        v34 = v21 + 1;
        v35 = *v21[1];
        if ( v35 <= 0x1Cu
          || (unsigned int)v35 - 42 > 0x11 && v35 != 84
          || (v34 = v21 + 2, v36 = *v21[2], v36 <= 0x1Cu)
          || (unsigned int)v36 - 42 > 0x11 && v36 != 84
          || (v34 = v21 + 3, v37 = *v21[3], v37 <= 0x1Cu)
          || (unsigned int)v37 - 42 > 0x11 && v37 != 84 )
        {
          v21 = v34;
          goto LABEL_29;
        }
        v21 += 4;
        if ( !--v22 )
        {
          v16 = v15 - v21;
          goto LABEL_72;
        }
      }
    }
    v42 = v16;
    v19 = *(unsigned __int8 ***)v11;
LABEL_102:
    if ( v42 == 2 )
      goto LABEL_148;
    if ( v42 == 3 )
    {
      v49 = **v19;
      if ( v49 > 0x15u && v49 != 84 )
        goto LABEL_107;
      ++v19;
LABEL_148:
      v50 = **v19;
      if ( v50 > 0x15u && v50 != 84 )
        goto LABEL_107;
      ++v19;
      goto LABEL_105;
    }
    if ( v42 != 1 )
      goto LABEL_20;
LABEL_105:
    v43 = **v19;
    if ( v43 <= 0x15u || v43 == 84 )
      goto LABEL_20;
LABEL_107:
    if ( v15 == v19 )
      goto LABEL_20;
    v21 = v13;
    if ( v17 )
    {
      v22 = v17;
      goto LABEL_28;
    }
LABEL_72:
    if ( v16 == 2 )
      goto LABEL_122;
    if ( v16 == 3 )
    {
      v45 = **v21;
      if ( v45 <= 0x1Cu || (unsigned int)v45 - 42 > 0x11 && v45 != 84 )
        goto LABEL_29;
      ++v21;
LABEL_122:
      v46 = **v21;
      if ( v46 <= 0x1Cu || (unsigned int)v46 - 42 > 0x11 && v46 != 84 )
        goto LABEL_29;
      ++v21;
      goto LABEL_75;
    }
    if ( v16 != 1 )
      goto LABEL_20;
LABEL_75:
    v38 = **v21;
    if ( v38 > 0x1Cu && ((unsigned int)v38 - 42 <= 0x11 || v38 == 84) )
      goto LABEL_20;
LABEL_29:
    if ( v15 != v21 )
      break;
LABEL_20:
    if ( v9 == ++v7 )
      return v8 ^ 1u;
  }
  if ( *(_DWORD *)(v3 + 8) != 2 )
    return 1;
  if ( !v17 )
    goto LABEL_111;
  v24 = &v13[4 * v17];
  while ( 2 )
  {
    v32 = **v13;
    if ( v32 <= 0x1Cu )
    {
      if ( v32 == 5 && *((_WORD *)*v13 + 1) == 34 )
        goto LABEL_48;
    }
    else if ( v32 == 63 || v32 == 84 )
    {
      goto LABEL_48;
    }
    v25 = v13[1];
    v26 = v13 + 1;
    v27 = *v25;
    if ( *v25 > 0x1Cu )
    {
      if ( v27 == 63 || v27 == 84 )
      {
LABEL_58:
        v13 = v26;
        goto LABEL_48;
      }
    }
    else if ( v27 == 5 && *((_WORD *)v25 + 1) == 34 )
    {
      goto LABEL_58;
    }
    v28 = v13[2];
    v26 = v13 + 2;
    v29 = *v28;
    if ( *v28 > 0x1Cu )
    {
      if ( v29 == 63 || v29 == 84 )
        goto LABEL_58;
      v30 = v13[3];
      v26 = v13 + 3;
      v31 = *v30;
      if ( *v30 > 0x1Cu )
        goto LABEL_83;
LABEL_42:
      if ( v31 == 5 && *((_WORD *)v30 + 1) == 34 )
        goto LABEL_58;
    }
    else
    {
      if ( v29 == 5 && *((_WORD *)v28 + 1) == 34 )
        goto LABEL_58;
      v30 = v13[3];
      v26 = v13 + 3;
      v31 = *v30;
      if ( *v30 <= 0x1Cu )
        goto LABEL_42;
LABEL_83:
      if ( v31 == 63 || v31 == 84 )
        goto LABEL_58;
    }
    v13 += 4;
    if ( v24 != v13 )
      continue;
    break;
  }
  v18 = v15 - v13;
LABEL_111:
  switch ( v18 )
  {
    case 2LL:
      goto LABEL_131;
    case 3LL:
      v47 = **v13;
      if ( v47 > 0x1Cu )
      {
        if ( v47 == 63 || v47 == 84 )
          goto LABEL_48;
      }
      else if ( v47 == 5 && *((_WORD *)*v13 + 1) == 34 )
      {
        goto LABEL_48;
      }
      ++v13;
LABEL_131:
      v48 = **v13;
      if ( v48 > 0x1Cu )
      {
        if ( v48 != 84 && v48 != 63 )
          goto LABEL_134;
      }
      else if ( v48 != 5 || *((_WORD *)*v13 + 1) != 34 )
      {
LABEL_134:
        ++v13;
LABEL_114:
        v44 = **v13;
        if ( v44 > 0x1Cu )
        {
          if ( v44 != 84 && v44 != 63 )
            return 1;
        }
        else if ( v44 != 5 || *((_WORD *)*v13 + 1) != 34 )
        {
          return 1;
        }
      }
LABEL_48:
      if ( v15 == v13 )
        return 1;
      goto LABEL_20;
    case 1LL:
      goto LABEL_114;
  }
  return 1;
}
