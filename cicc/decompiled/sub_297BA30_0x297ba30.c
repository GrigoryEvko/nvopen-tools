// Function: sub_297BA30
// Address: 0x297ba30
//
bool __fastcall sub_297BA30(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  __int64 v5; // r13

  switch ( a3 )
  {
    case 2:
      v4 = sub_D95540(*(_QWORD *)(a1 + 8));
      if ( v4 != sub_D95540(*(_QWORD *)(a2 + 8)) )
        return 0;
      break;
    case 3:
      v5 = sub_D95540(*(_QWORD *)(a1 + 56));
      if ( v5 != sub_D95540(*(_QWORD *)(a2 + 56)) || *(_QWORD *)(a2 + 32) == *(_QWORD *)(a1 + 32) )
        return 0;
      return *(_DWORD *)a2 == *(_DWORD *)a1;
    case 1:
      break;
    default:
      return 0;
  }
  if ( *(_QWORD *)(a2 + 32) == *(_QWORD *)(a1 + 32) )
    return 0;
  return *(_DWORD *)a2 == *(_DWORD *)a1;
}
