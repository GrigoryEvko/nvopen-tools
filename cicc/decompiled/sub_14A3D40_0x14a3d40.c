// Function: sub_14A3D40
// Address: 0x14a3d40
//
__int64 __fastcall sub_14A3D40(__int64 a1, __int64 a2, __int64 a3)
{
  if ( !*(_QWORD *)(a2 + 16) )
    sub_4263D6(a1, a2, a3);
  (*(void (**)(void))(a2 + 24))();
  return a1;
}
