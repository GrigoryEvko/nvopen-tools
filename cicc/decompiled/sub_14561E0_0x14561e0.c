// Function: sub_14561E0
// Address: 0x14561e0
//
__int64 __fastcall sub_14561E0(__int64 a1)
{
  __int64 v1; // rbx

  if ( *(_WORD *)(a1 + 24) != 4 )
    return 0;
  if ( *(_QWORD *)(a1 + 40) == 2 && sub_1456170(**(_QWORD **)(a1 + 32)) )
  {
    v1 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
    if ( *(_WORD *)(v1 + 24) == 5 && *(_QWORD *)(v1 + 40) == 2 && sub_1456170(**(_QWORD **)(v1 + 32)) )
      return *(_QWORD *)(*(_QWORD *)(v1 + 32) + 8LL);
  }
  return 0;
}
