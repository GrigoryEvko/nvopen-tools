// Function: sub_2B27770
// Address: 0x2b27770
//
__int64 __fastcall sub_2B27770(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rdi
  bool v4; // al
  unsigned __int8 v5; // dl
  __int64 v6; // rdi
  bool v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int8 v11; // r12
  __int64 v12; // r14
  _BYTE *v13; // r13
  _BYTE *v14; // r15
  char v15; // al
  __int64 result; // rax
  __int64 v17; // r13
  __int64 v18; // r13
  __int64 v19; // rax
  _BYTE *v20; // rdx
  _BYTE *v21; // rax
  __int16 v22; // ax
  int v23; // eax
  _BYTE *v24; // rdx
  _BYTE *v25; // rax
  __int16 v26; // ax
  int v27; // eax
  _BYTE *v28; // r12
  int v29; // ebx
  __int64 v30; // rbx
  _BYTE *v31; // rdx
  _BYTE *v32; // rax
  __int16 v33; // ax
  _BYTE *v34; // rdx
  _BYTE *v35; // rax
  __int16 v36; // ax
  _BYTE *v37; // r12
  _BYTE *v38; // [rsp-40h] [rbp-40h]

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 0;
  switch ( v1 )
  {
    case '*':
      return 1;
    case '.':
      return 2;
    case '9':
      return 4;
  }
  v3 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  v4 = sub_BCAC40(v3, 1);
  v5 = *(_BYTE *)a1;
  if ( !v4 )
    goto LABEL_26;
  if ( v5 == 57 )
    return 4;
  if ( v5 == 86 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    if ( *(_QWORD *)(*(_QWORD *)(a1 - 96) + 8LL) != v6 || **(_BYTE **)(a1 - 32) > 0x15u )
      goto LABEL_11;
    if ( !sub_AC30F0(*(_QWORD *)(a1 - 32)) )
    {
      v5 = *(_BYTE *)a1;
      goto LABEL_26;
    }
    return 4;
  }
LABEL_26:
  result = 3;
  if ( v5 == 58 )
    return result;
  v6 = *(_QWORD *)(a1 + 8);
LABEL_11:
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v7 = sub_BCAC40(v6, 1);
  v11 = *(_BYTE *)a1;
  if ( v7 )
  {
    if ( v11 == 58 )
      return 3;
    if ( v11 != 86 )
      goto LABEL_34;
    v12 = *(_QWORD *)(a1 - 96);
    if ( *(_QWORD *)(v12 + 8) != *(_QWORD *)(a1 + 8) || **(_BYTE **)(a1 - 64) > 0x15u )
    {
      if ( !sub_988010(a1) )
        goto LABEL_18;
      goto LABEL_59;
    }
    if ( sub_AD7A80(*(_BYTE **)(a1 - 64), 1, v8, v9, v10) )
      return 3;
    v11 = *(_BYTE *)a1;
  }
LABEL_34:
  if ( v11 == 59 )
    return 5;
  if ( v11 == 43 )
    return 10;
  result = 11;
  if ( v11 != 47 )
  {
    if ( v11 == 85 )
    {
      v17 = *(_QWORD *)(a1 - 32);
      if ( !v17 )
      {
        if ( !sub_988010(a1) )
          goto LABEL_72;
        goto LABEL_162;
      }
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a1 + 80) && *(_DWORD *)(v17 + 36) == 237 )
        return 13;
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a1 + 80) && *(_DWORD *)(v17 + 36) == 248 )
        return 12;
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a1 + 80) && *(_DWORD *)(v17 + 36) == 235 )
        return 15;
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a1 + 80) && *(_DWORD *)(v17 + 36) == 246 )
        return 14;
      if ( !sub_988010(a1) )
        goto LABEL_72;
LABEL_48:
      if ( !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(a1 + 80) )
      {
        if ( *(_DWORD *)(v17 + 36) == 329 )
          return 7;
        goto LABEL_51;
      }
LABEL_162:
      BUG();
    }
    if ( !sub_988010(a1) )
    {
LABEL_51:
      if ( v11 != 86 )
        goto LABEL_62;
      v12 = *(_QWORD *)(a1 - 96);
LABEL_18:
      v13 = *(_BYTE **)(a1 - 32);
      v14 = *(_BYTE **)(a1 - 64);
      if ( *(_BYTE *)v12 != 82 )
        goto LABEL_19;
      v24 = *(_BYTE **)(v12 - 64);
      v25 = *(_BYTE **)(v12 - 32);
      if ( v14 == v24 && v13 == v25 )
      {
        v26 = *(_WORD *)(v12 + 2);
      }
      else
      {
        if ( v14 != v25 || v13 != v24 )
          goto LABEL_90;
        v26 = *(_WORD *)(v12 + 2);
        if ( v14 != v24 )
        {
          v27 = sub_B52870(*(_WORD *)(v12 + 2) & 0x3F);
          goto LABEL_100;
        }
      }
      v27 = v26 & 0x3F;
LABEL_100:
      if ( (unsigned int)(v27 - 38) <= 1 )
        return 7;
      v11 = *(_BYTE *)a1;
LABEL_62:
      if ( v11 <= 0x1Cu )
      {
LABEL_63:
        if ( *(_BYTE *)a1 != 86 )
          return 0;
        v12 = *(_QWORD *)(a1 - 96);
        v13 = *(_BYTE **)(a1 - 32);
        v14 = *(_BYTE **)(a1 - 64);
LABEL_19:
        v15 = *(_BYTE *)v12;
        if ( *(_BYTE *)v12 > 0x1Cu && (unsigned __int8)(v15 - 82) <= 1u )
        {
          if ( v14 == *(_BYTE **)(v12 - 64) )
          {
            v37 = *(_BYTE **)(v12 - 32);
            if ( *v37 > 0x1Cu )
            {
              v29 = sub_B53900(v12);
              if ( *v13 != 90 )
                return 0;
              goto LABEL_145;
            }
          }
          v28 = *(_BYTE **)(v12 - 64);
          if ( *v28 > 0x1Cu && v13 == *(_BYTE **)(v12 - 32) )
          {
            v29 = sub_B53900(v12);
            if ( *v14 != 90 || !sub_B46220((__int64)v28, (__int64)v14) )
              return 0;
            goto LABEL_111;
          }
        }
        if ( *v14 != 90 )
          return 0;
        if ( *v13 != 90 )
          return 0;
        if ( (unsigned __int8)(v15 - 82) > 1u )
          return 0;
        v38 = *(_BYTE **)(v12 - 64);
        if ( *v38 <= 0x1Cu )
          return 0;
        v37 = *(_BYTE **)(v12 - 32);
        if ( *v37 <= 0x1Cu )
          return 0;
        v29 = sub_B53900(v12);
        if ( !sub_B46220((__int64)v38, (__int64)v14) )
          return 0;
LABEL_145:
        if ( !sub_B46220((__int64)v37, (__int64)v13) )
          return 0;
LABEL_111:
        v30 = (unsigned int)(v29 - 34);
        if ( (unsigned int)v30 <= 7 )
          return *(unsigned int *)&asc_439FE20[4 * v30];
        return 0;
      }
      if ( v11 != 85 )
      {
        if ( v11 != 86 )
          goto LABEL_78;
        v12 = *(_QWORD *)(a1 - 96);
        v13 = *(_BYTE **)(a1 - 32);
        v14 = *(_BYTE **)(a1 - 64);
        if ( *(_BYTE *)v12 != 82 )
          goto LABEL_19;
LABEL_90:
        v20 = *(_BYTE **)(v12 - 64);
        v21 = *(_BYTE **)(v12 - 32);
        if ( v14 == v20 && v13 == v21 )
        {
          v22 = *(_WORD *)(v12 + 2);
        }
        else
        {
          if ( v14 != v21 || v13 != v20 )
          {
LABEL_115:
            v31 = *(_BYTE **)(v12 - 64);
            v32 = *(_BYTE **)(v12 - 32);
            if ( v14 == v31 && v13 == v32 )
            {
              v33 = *(_WORD *)(v12 + 2);
LABEL_118:
              if ( (v33 & 0x3Fu) - 34 > 1 )
                goto LABEL_126;
              return 9;
            }
            if ( v14 != v32 || v13 != v31 )
              goto LABEL_126;
            v33 = *(_WORD *)(v12 + 2);
            if ( v14 == v31 )
              goto LABEL_118;
            if ( (unsigned int)sub_B52870(*(_WORD *)(v12 + 2) & 0x3F) - 34 <= 1 )
              return 9;
            v11 = *(_BYTE *)a1;
            if ( *(_BYTE *)a1 <= 0x1Cu )
              goto LABEL_63;
LABEL_78:
            if ( v11 == 85 )
            {
              v19 = *(_QWORD *)(a1 - 32);
              if ( !v19
                || *(_BYTE *)v19
                || *(_QWORD *)(v19 + 24) != *(_QWORD *)(a1 + 80)
                || (*(_BYTE *)(v19 + 33) & 0x20) == 0
                || *(_DWORD *)(v19 + 36) != 366 )
              {
                return 0;
              }
              return 8;
            }
            if ( v11 != 86 )
              return 0;
            v12 = *(_QWORD *)(a1 - 96);
            v13 = *(_BYTE **)(a1 - 32);
            v14 = *(_BYTE **)(a1 - 64);
            if ( *(_BYTE *)v12 != 82 )
              goto LABEL_19;
LABEL_126:
            v34 = *(_BYTE **)(v12 - 64);
            v35 = *(_BYTE **)(v12 - 32);
            if ( v34 == v14 && v35 == v13 )
            {
              v36 = *(_WORD *)(v12 + 2);
            }
            else
            {
              if ( v35 != v14 || v34 != v13 )
                goto LABEL_19;
              v36 = *(_WORD *)(v12 + 2);
              if ( v34 != v14 )
              {
                if ( (unsigned int)sub_B52870(*(_WORD *)(v12 + 2) & 0x3F) - 36 > 1 )
                  goto LABEL_63;
                return 8;
              }
            }
            if ( (v36 & 0x3Fu) - 36 > 1 )
              goto LABEL_19;
            return 8;
          }
          v22 = *(_WORD *)(v12 + 2);
          if ( v14 != v20 )
          {
            v23 = sub_B52870(*(_WORD *)(v12 + 2) & 0x3F);
            goto LABEL_94;
          }
        }
        v23 = v22 & 0x3F;
LABEL_94:
        if ( (unsigned int)(v23 - 40) <= 1 )
          return 6;
        v11 = *(_BYTE *)a1;
        if ( *(_BYTE *)a1 <= 0x1Cu )
          goto LABEL_63;
        if ( v11 == 85 )
        {
          v18 = *(_QWORD *)(a1 - 32);
          goto LABEL_75;
        }
        if ( v11 != 86 )
          goto LABEL_78;
        v12 = *(_QWORD *)(a1 - 96);
        v13 = *(_BYTE **)(a1 - 32);
        v14 = *(_BYTE **)(a1 - 64);
        if ( *(_BYTE *)v12 != 82 )
          goto LABEL_19;
        goto LABEL_115;
      }
LABEL_72:
      v18 = *(_QWORD *)(a1 - 32);
      if ( v18 )
      {
        if ( !*(_BYTE *)v18
          && *(_QWORD *)(v18 + 24) == *(_QWORD *)(a1 + 80)
          && (*(_BYTE *)(v18 + 33) & 0x20) != 0
          && *(_DWORD *)(v18 + 36) == 330 )
        {
          return 6;
        }
LABEL_75:
        if ( v18
          && !*(_BYTE *)v18
          && *(_QWORD *)(v18 + 24) == *(_QWORD *)(a1 + 80)
          && (*(_BYTE *)(v18 + 33) & 0x20) != 0
          && *(_DWORD *)(v18 + 36) == 365 )
        {
          return 9;
        }
        goto LABEL_78;
      }
      goto LABEL_78;
    }
LABEL_59:
    v17 = *(_QWORD *)(a1 - 32);
    if ( !v17 )
      goto LABEL_162;
    goto LABEL_48;
  }
  return result;
}
