// Function: sub_1041420
// Address: 0x1041420
//
char __fastcall sub_1041420(__int64 a1, __int64 a2, __int64 a3)
{
  if ( a2 == a3 )
    return 1;
  if ( a3 == *(_QWORD *)(a1 + 128) )
    return 0;
  if ( *(_QWORD *)(a2 + 64) == *(_QWORD *)(a3 + 64) )
    return sub_1041270(a1, a2, a3);
  return sub_B19720(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 64), *(_QWORD *)(a3 + 64));
}
