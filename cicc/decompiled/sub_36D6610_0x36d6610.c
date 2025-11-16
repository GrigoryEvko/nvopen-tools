// Function: sub_36D6610
// Address: 0x36d6610
//
__int64 __fastcall sub_36D6610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int16 v5; // ax
  unsigned int v6; // r8d

  v5 = *(_WORD *)(a2 + 68);
  if ( v5 > 0x128Eu )
  {
    LOBYTE(a5) = (unsigned __int16)(v5 - 4753) <= 9u;
    return a5;
  }
  else
  {
    v6 = 1;
    if ( v5 <= 0x122Eu )
      LOBYTE(v6) = (unsigned __int16)(v5 - 4643) <= 9u;
    return v6;
  }
}
