// Function: sub_1FDDF40
// Address: 0x1fddf40
//
__int64 __fastcall sub_1FDDF40(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx

  v2 = *(_QWORD *)(a2 + 40);
  v3 = *(_QWORD *)(a2 + 8);
  if ( !v3 )
    return 1;
  while ( v2 != sub_1648700(v3)[5] )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return 1;
  }
  return 0;
}
