// Function: sub_DFBCC0
// Address: 0xdfbcc0
//
__int64 __fastcall sub_DFBCC0(unsigned __int8 *a1)
{
  int v1; // edx
  unsigned int v2; // r8d
  unsigned __int8 *v3; // rdi
  unsigned __int8 *v4; // rdx
  unsigned __int8 v5; // al
  __int64 v7; // rax
  char *v8; // rdx
  unsigned __int8 v9; // al
  unsigned __int8 *v10; // rax
  int v11; // eax
  unsigned __int8 *v12; // rax
  int v13; // eax

  if ( a1 )
  {
    v1 = *a1;
    if ( v1 != 74 )
    {
      if ( (unsigned int)(v1 - 29) > 0x2D )
      {
        if ( v1 != 75 )
          return 0;
        goto LABEL_6;
      }
      if ( v1 != 67 )
      {
        v2 = 0;
        if ( (unsigned __int8)(v1 - 68) > 1u )
          return v2;
LABEL_6:
        if ( (a1[7] & 0x40) != 0 )
          v3 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        else
          v3 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v4 = *(unsigned __int8 **)v3;
        v2 = 0;
        v5 = **(_BYTE **)v3;
        if ( v5 <= 0x1Cu )
          return v2;
        v2 = 1;
        if ( v5 == 61 )
          return v2;
        v2 = 0;
        if ( v5 != 85 )
          return v2;
        v12 = (unsigned __int8 *)*((_QWORD *)v4 - 4);
        if ( !v12 )
          return v2;
        v2 = *v12;
        if ( !(_BYTE)v2 )
        {
          if ( *((_QWORD *)v12 + 3) != *((_QWORD *)v4 + 10) || (v12[33] & 0x20) == 0 )
            return v2;
          v13 = *((_DWORD *)v12 + 9);
          if ( v13 != 228 )
          {
            if ( v13 == 227 )
              return 3;
            return v2;
          }
          return 2;
        }
        return 0;
      }
    }
    v7 = *((_QWORD *)a1 + 2);
    v2 = 0;
    if ( !v7 )
      return v2;
    if ( *(_QWORD *)(v7 + 8) )
      return v2;
    v8 = *(char **)(v7 + 24);
    v9 = *v8;
    if ( (unsigned __int8)*v8 <= 0x1Cu )
      return v2;
    v2 = 1;
    if ( v9 == 62 )
      return v2;
    v2 = 0;
    if ( v9 != 85 )
      return v2;
    v10 = (unsigned __int8 *)*((_QWORD *)v8 - 4);
    if ( !v10 )
      return v2;
    v2 = *v10;
    if ( !(_BYTE)v2 )
    {
      if ( *((_QWORD *)v10 + 3) != *((_QWORD *)v8 + 10) || (v10[33] & 0x20) == 0 )
        return v2;
      v11 = *((_DWORD *)v10 + 9);
      if ( v11 != 230 )
      {
        if ( v11 == 229 )
          return 3;
        return v2;
      }
      return 2;
    }
  }
  return 0;
}
