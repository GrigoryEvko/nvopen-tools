// Function: sub_386ECF0
// Address: 0x386ecf0
//
bool __fastcall sub_386ECF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v8; // r15

  v8 = *(_BYTE *)(sub_1456040(a3) + 8) == 15;
  if ( (*(_BYTE *)(sub_1456040(a5) + 8) == 15) != v8 )
    return *(_BYTE *)(sub_1456040(a3) + 8) == 15;
  if ( a4 != a2 )
    return sub_386EC30(a2, a4, *a1) != a2;
  if ( !sub_1456260(a3) )
    return sub_1456260(a5);
  sub_1456260(a5);
  return 0;
}
