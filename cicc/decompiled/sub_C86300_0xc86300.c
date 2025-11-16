// Function: sub_C86300
// Address: 0xc86300
//
__int64 sub_C86300()
{
  unsigned int v0; // eax
  unsigned int v1; // r12d
  char *v3; // rax
  char *v4; // rbx
  size_t v5; // rax
  __int64 v6; // rax

  v0 = sub_C862F0();
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
      if ( *(_DWORD *)v4 == 1769172577 )
        return v1;
LABEL_7:
      if ( *(_DWORD *)v4 == 1953921138 )
        return v1;
      if ( v5 != 4 )
      {
        v6 = (__int64)&v4[v5 - 5];
        if ( *(_DWORD *)v6 == 1869377379 && *(_BYTE *)(v6 + 4) == 114 )
          return v1;
      }
      return 0;
    case 6uLL:
      if ( *(_DWORD *)v4 == 2003269987 && *((_WORD *)v4 + 2) == 28265 )
        return v1;
      break;
    case 5uLL:
      if ( *(_DWORD *)v4 == 1970170220 && v4[4] == 120 )
        return v1;
      goto LABEL_19;
    default:
      if ( v5 <= 5 )
        goto LABEL_22;
      break;
  }
  if ( *(_DWORD *)v4 == 1701995379 && *((_WORD *)v4 + 2) == 28261 )
    return v1;
LABEL_19:
  if ( (*(_DWORD *)v4 != 1919251576 || v4[4] != 109) && (*(_DWORD *)v4 != 808547446 || v4[4] != 48) )
  {
LABEL_22:
    if ( v5 <= 3 )
      return 0;
    goto LABEL_7;
  }
  return v1;
}
