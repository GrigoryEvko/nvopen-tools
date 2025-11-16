// Function: sub_109F820
// Address: 0x109f820
//
__int64 __fastcall sub_109F820(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // rdi
  unsigned int v6; // r14d
  char *v7; // r12

  v4 = *(char **)a1;
  if ( v4 && (v6 = sub_109F4E0(v4, a2, a3)) != 0 )
  {
    if ( *(_BYTE *)(a1 + 8) || *(_WORD *)(a1 + 10) != 1 )
    {
      v7 = (char *)(a1 + 8);
      sub_109E080(a2 + 8, v7);
      if ( v6 == 2 )
        sub_109E080(a3 + 8, v7);
    }
  }
  else
  {
    return 0;
  }
  return v6;
}
