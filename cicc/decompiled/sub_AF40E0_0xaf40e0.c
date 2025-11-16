// Function: sub_AF40E0
// Address: 0xaf40e0
//
__int64 __fastcall sub_AF40E0(int a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // r12

  v2 = sub_B97910(16, 0, a2);
  v3 = v2;
  if ( v2 )
    sub_B971C0(v2, a1, 30, a2, 0, 0, 0, 0);
  if ( !a2 )
    BUG();
  if ( a2 == 1 )
    sub_B95A20(v3);
  return v3;
}
