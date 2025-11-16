// Function: sub_16F5560
// Address: 0x16f5560
//
char *__fastcall sub_16F5560(char *a1, __int64 a2)
{
  char *result; // rax
  bool v3; // r8
  bool v4; // r10
  char v5; // dl
  char v6; // r9
  char v7; // cl
  char v8; // r13
  char *v9; // r14
  char v10; // r12
  char v11; // r12
  bool v12; // bl
  char v13; // dl

  result = a1;
  v3 = a2 == 4;
  if ( a2 == 2 )
  {
    if ( *(_WORD *)a1 == 13686 )
    {
      v9 = (char *)&unk_3F88AB3;
      v7 = 0;
      goto LABEL_24;
    }
    goto LABEL_8;
  }
  if ( a2 == 3 )
  {
    if ( *(_WORD *)a1 != 13686 || (v9 = "v5te", a1[2] != 101) )
    {
      if ( *(_WORD *)a1 != 13942 || (v9 = "v6", a1[2] != 106) )
      {
        v4 = 0;
        if ( *(_WORD *)a1 == 13942 && a1[2] == 109 )
        {
          v7 = 1;
          v8 = 1;
          v5 = 0;
          v9 = "v6-m";
          v6 = 1;
        }
        else
        {
          v5 = 1;
          v6 = 0;
          v7 = 1;
          v8 = 0;
          v9 = (char *)&unk_3F88AB3;
        }
        goto LABEL_10;
      }
    }
    v7 = 1;
LABEL_24:
    v6 = 1;
    v8 = 1;
    v4 = a2 == 5;
    v5 = 0;
LABEL_10:
    v10 = v7 & v5;
    goto LABEL_11;
  }
  if ( a2 != 4 )
  {
LABEL_8:
    v7 = 0;
    v4 = a2 == 5;
    v6 = 0;
    v8 = 0;
    v5 = 1;
    v9 = (char *)&unk_3F88AB3;
    goto LABEL_9;
  }
  if ( *(_DWORD *)a1 == 1818769014 )
  {
    v4 = 0;
    v7 = 0;
    v6 = 1;
    v5 = 0;
    v8 = 1;
    v9 = "v6k";
LABEL_9:
    if ( ((unsigned __int8)v5 & v4) != 0 )
    {
      if ( *(_DWORD *)a1 == 762525302 && a1[4] == 109 )
      {
        v6 = v5 & v4;
        v8 = 1;
        v12 = a2 == 2;
        v9 = "v6-m";
        v5 = 0;
      }
      else
      {
        v11 = v3 & v5;
        v12 = a2 == 2;
        if ( (v3 & (unsigned __int8)v5) != 0 )
          goto LABEL_27;
      }
LABEL_13:
      if ( ((unsigned __int8)v5 & v12) != 0 )
      {
        if ( *(_WORD *)a1 == 14198 )
          return "v7-a";
        goto LABEL_15;
      }
      goto LABEL_14;
    }
    goto LABEL_10;
  }
  v4 = 0;
  v7 = 0;
  if ( *(_DWORD *)a1 == 1836267126 )
  {
    v5 = 0;
    v6 = 1;
    v8 = 1;
    v9 = "v6-m";
    goto LABEL_12;
  }
  v10 = 0;
  v5 = 1;
  v6 = 0;
  v8 = 0;
  v9 = (char *)&unk_3F88AB3;
LABEL_11:
  if ( v10 )
  {
    v12 = a2 == 2;
    if ( *(_WORD *)a1 == 13942 && a1[2] == 122 )
    {
      v6 = v10;
      v5 = 0;
      v8 = 1;
      v9 = "v6kz";
      goto LABEL_14;
    }
    goto LABEL_13;
  }
LABEL_12:
  v11 = v5 & v3;
  v12 = a2 == 2;
  if ( ((unsigned __int8)v5 & v3) == 0 )
    goto LABEL_13;
LABEL_27:
  if ( *(_DWORD *)a1 == 1803171446 )
  {
    v6 = v11;
    v5 = 0;
    v8 = 1;
    v9 = "v6kz";
    goto LABEL_15;
  }
LABEL_14:
  if ( ((unsigned __int8)v7 & (unsigned __int8)v5) == 0 )
  {
LABEL_15:
    if ( ((unsigned __int8)v5 & v3) == 0 )
      goto LABEL_16;
    if ( *(_DWORD *)a1 != 1818769270 )
    {
      if ( v6 )
      {
        v6 = v12 & v5;
        if ( (v12 & (unsigned __int8)v5) == 0 )
          return v9;
        goto LABEL_42;
      }
      goto LABEL_73;
    }
    return "v7-a";
  }
  if ( *(_WORD *)a1 == 14198 && a1[2] == 97 )
    return "v7-a";
LABEL_16:
  if ( v6 )
    return v9;
  if ( v7 )
  {
    if ( *(_WORD *)a1 == 14198 && a1[2] == 108 )
      return "v7-a";
    if ( *(_WORD *)a1 == 14198 && a1[2] == 114 )
      return "v7-r";
    if ( *(_WORD *)a1 == 14198 && a1[2] == 109 )
      return "v7-m";
    v6 = v8;
    v5 = v8 ^ 1;
LABEL_73:
    if ( !v5 )
      goto LABEL_33;
  }
  if ( a2 == 4 )
  {
    if ( *(_DWORD *)a1 == 1835349878 )
      return "v7e-m";
LABEL_33:
    if ( v6 )
      return v9;
    if ( v7 )
    {
      if ( *(_WORD *)a1 == 14454 && a1[2] == 97 || *(_WORD *)a1 == 14454 && a1[2] == 108 )
        return "v8-a";
      if ( a2 == 7 )
      {
        if ( *(_DWORD *)a1 == 1668440417 && *((_WORD *)a1 + 2) == 13928 && a1[6] == 52 )
          return "v8-a";
        goto LABEL_46;
      }
    }
    else
    {
      v13 = 0;
      if ( a2 == 7 )
        goto LABEL_36;
    }
LABEL_45:
    if ( v4 )
    {
      if ( *(_DWORD *)a1 == 913142369 && a1[4] == 52 )
        return "v8-a";
      if ( *(_DWORD *)a1 == 825112694 && a1[4] == 97 )
        return "v8.1-a";
      if ( *(_DWORD *)a1 == 841889910 && a1[4] == 97 )
        return "v8.2-a";
      if ( v8 )
        return v9;
      if ( *(_DWORD *)a1 == 858667126 && a1[4] == 97 )
        return "v8.3-a";
      if ( *(_DWORD *)a1 == 875444342 && a1[4] == 97 )
        return "v8.4-a";
    }
    goto LABEL_46;
  }
  if ( !v12 )
    goto LABEL_33;
LABEL_42:
  if ( *(_WORD *)a1 == 14454 )
    return "v8-a";
  if ( a2 != 7 )
  {
    if ( v6 )
      return v9;
    goto LABEL_45;
  }
  v13 = v7;
  v7 = v6;
LABEL_36:
  if ( *(_DWORD *)a1 == 1668440417 && *((_WORD *)a1 + 2) == 13928 && a1[6] == 52 )
    return "v8-a";
  if ( v7 )
    return v9;
  v7 = v13;
LABEL_46:
  if ( v7 )
  {
    if ( *(_WORD *)a1 == 14454 )
    {
      result = "v8-r";
      if ( a1[2] != 114 )
        return a1;
    }
  }
  else if ( a2 == 8 )
  {
    if ( *(_QWORD *)a1 == 0x657361622E6D3876LL )
    {
      return "v8-m.base";
    }
    else if ( *(_QWORD *)a1 == 0x6E69616D2E6D3876LL )
    {
      return "v8-m.main";
    }
  }
  return result;
}
