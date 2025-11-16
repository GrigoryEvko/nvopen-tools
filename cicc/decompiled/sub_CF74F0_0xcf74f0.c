// Function: sub_CF74F0
// Address: 0xcf74f0
//
char __fastcall sub_CF74F0(unsigned __int8 *a1)
{
  int v1; // ecx
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int8 v4; // cl

  v1 = *a1;
  if ( (unsigned __int8)v1 <= 0x1Cu )
  {
    if ( (_BYTE)v1 == 5 )
    {
      LOBYTE(v2) = *((_WORD *)a1 + 1) == 48;
      return v2;
    }
LABEL_7:
    LOBYTE(v2) = 0;
    return v2;
  }
  if ( (unsigned __int8)(v1 - 34) <= 0x33u )
  {
    v3 = 0x8000000000041LL;
    if ( _bittest64(&v3, (unsigned int)(v1 - 34)) )
    {
      if ( !sub_98AB90((__int64)a1, 1) )
      {
        LODWORD(v2) = sub_B4A0B0(a1) ^ 1;
        return v2;
      }
      goto LABEL_7;
    }
  }
  v4 = v1 - 61;
  LOBYTE(v2) = 0;
  if ( v4 <= 0x20u )
    return (0x120010001uLL >> v4) & 1;
  return v2;
}
