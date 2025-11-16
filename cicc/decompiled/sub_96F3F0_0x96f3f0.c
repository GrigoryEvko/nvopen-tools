// Function: sub_96F3F0
// Address: 0x96f3f0
//
__int64 __fastcall sub_96F3F0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v5; // rdi
  unsigned int v8; // r15d
  __int64 v9; // rdi

  v5 = *(_QWORD *)(a1 + 8);
  if ( v5 == a2 )
    return a1;
  v8 = sub_BCB060(v5);
  v9 = 38;
  if ( v8 <= (unsigned int)sub_BCB060(a2) )
  {
    v9 = 40;
    if ( !a3 )
      v9 = 39;
  }
  return sub_96F480(v9, a1, a2, a4);
}
