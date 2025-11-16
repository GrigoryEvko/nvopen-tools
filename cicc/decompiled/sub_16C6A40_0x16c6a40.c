// Function: sub_16C6A40
// Address: 0x16c6a40
//
__int64 sub_16C6A40()
{
  unsigned int v0; // eax
  unsigned int v1; // r12d
  char *v3; // rax
  char *v4; // rbx
  size_t v5; // rax
  bool v6; // cl
  __int64 v7; // rax
  bool v8; // dl
  int v9; // esi

  v0 = sub_16C6A30();
  if ( !(_BYTE)v0 )
    return 0;
  v1 = v0;
  v3 = getenv("TERM");
  v4 = v3;
  if ( !v3 )
    return 0;
  v5 = strlen(v3);
  switch ( v5 )
  {
    case 4uLL:
      v8 = *(_DWORD *)v4 == 1769172577;
      v6 = v8;
LABEL_18:
      if ( v8 )
        return v1;
LABEL_19:
      if ( v5 > 4 )
        goto LABEL_10;
      if ( v5 == 4 && *(_DWORD *)v4 == 1953921138 )
        return v1;
      return 0;
    case 6uLL:
      if ( *(_DWORD *)v4 != 2003269987 || (v9 = 0, *((_WORD *)v4 + 2) != 28265) )
        v9 = 1;
      v8 = v9 == 0;
      v6 = v9 == 0;
      if ( v9 )
        goto LABEL_9;
      goto LABEL_18;
    case 5uLL:
      if ( *(_DWORD *)v4 == 1970170220 )
      {
        if ( v4[4] == 120 )
          return v1;
        v6 = 0;
      }
      else
      {
        v6 = 0;
      }
      goto LABEL_10;
  }
  v6 = 0;
  if ( v5 <= 5 )
    goto LABEL_19;
LABEL_9:
  if ( *(_DWORD *)v4 == 1701995379 && *((_WORD *)v4 + 2) == 28261 )
    return v1;
LABEL_10:
  if ( (*(_DWORD *)v4 != 1919251576 || v4[4] != 109)
    && (*(_DWORD *)v4 != 808547446 || v4[4] != 48)
    && *(_DWORD *)v4 != 1953921138 )
  {
    if ( v6 )
      return 0;
    v7 = (__int64)&v4[v5 - 5];
    if ( *(_DWORD *)v7 != 1869377379 || *(_BYTE *)(v7 + 4) != 114 )
      return 0;
  }
  return v1;
}
