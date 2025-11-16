// Function: sub_2491430
// Address: 0x2491430
//
__int64 __fastcall sub_2491430(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  if ( !a3 )
  {
    a3 = *(_QWORD *)(a1 + 96);
    if ( !a3 )
      return sub_B45150(a2, a4);
  }
  sub_B99FD0(a2, 3u, a3);
  return sub_B45150(a2, a4);
}
