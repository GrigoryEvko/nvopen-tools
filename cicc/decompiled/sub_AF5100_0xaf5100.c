// Function: sub_AF5100
// Address: 0xaf5100
//
__int64 __fastcall sub_AF5100(__int64 a1, char a2)
{
  if ( a2 )
    sub_AF50C0(a1);
  *(_DWORD *)(a1 + 144) = 0;
  return sub_B92F50(a1 + 8, 0);
}
