// Function: sub_AA6160
// Address: 0xaa6160
//
__int64 __fastcall sub_AA6160(__int64 a1, __int64 a2)
{
  if ( a2 == a1 + 48 )
    return sub_AA60B0(a1);
  if ( !a2 )
    BUG();
  return *(_QWORD *)(a2 + 40);
}
