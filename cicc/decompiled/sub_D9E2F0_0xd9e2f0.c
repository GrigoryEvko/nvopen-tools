// Function: sub_D9E2F0
// Address: 0xd9e2f0
//
unsigned __int64 __fastcall sub_D9E2F0(__int64 a1, __int64 a2)
{
  if ( !a2 )
    BUG();
  return sub_939680(*(_QWORD **)(a2 + 8), *(_QWORD *)(a2 + 8) + 4LL * *(_QWORD *)(a2 + 16));
}
