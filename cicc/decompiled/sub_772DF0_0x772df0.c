// Function: sub_772DF0
// Address: 0x772df0
//
void __fastcall sub_772DF0(__int64 a1, __int64 a2, __int64 a3)
{
  FILE *v3; // rsi
  char v5; // al

  v3 = (FILE *)(a2 + 28);
  v5 = *(_BYTE *)(a3 + 132) & 0x20;
  if ( (*(_BYTE *)(a1 + 8) & 8) != 0 )
  {
    if ( !v5 )
    {
      sub_67E440(0xA8Bu, v3, *(_DWORD *)(a1 + 8) >> 8, (_QWORD *)(a3 + 96));
      sub_770D30(a3);
    }
  }
  else if ( !v5 )
  {
    sub_6855B0(0xACCu, v3, (_QWORD *)(a3 + 96));
    sub_770D30(a3);
  }
}
