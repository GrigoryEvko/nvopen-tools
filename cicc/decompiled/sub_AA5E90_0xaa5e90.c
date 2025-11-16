// Function: sub_AA5E90
// Address: 0xaa5e90
//
bool __fastcall sub_AA5E90(__int64 a1)
{
  __int64 v1; // rax

  v1 = sub_AA4FF0(a1);
  if ( !v1 )
    BUG();
  return *(_BYTE *)(v1 - 24) == 95;
}
