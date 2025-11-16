// Function: sub_222DC80
// Address: 0x222dc80
//
void __fastcall sub_222DC80(__int64 a1, int a2)
{
  if ( !*(_QWORD *)(a1 + 232) )
    a2 |= 1u;
  *(_DWORD *)(a1 + 32) = a2;
  if ( (*(_DWORD *)(a1 + 28) & a2) != 0 )
    sub_426A1E((__int64)"basic_ios::clear");
}
