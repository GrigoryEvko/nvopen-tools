// Function: sub_349D790
// Address: 0x349d790
//
bool __fastcall sub_349D790(__int64 a1, __int64 a2)
{
  int v2; // eax
  bool result; // al

  v2 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 != *(_DWORD *)a2 )
    return 0;
  if ( v2 != 2 )
  {
    if ( v2 <= 2 )
    {
      if ( v2 != 1 )
        goto LABEL_15;
    }
    else if ( v2 != 3 )
    {
      if ( v2 == 4 )
      {
        result = 0;
        if ( *(_DWORD *)(a1 + 8) == *(_DWORD *)(a2 + 8) )
          return *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 16);
        return result;
      }
LABEL_15:
      BUG();
    }
    return *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8);
  }
  result = 0;
  if ( *(_DWORD *)(a1 + 8) == *(_DWORD *)(a2 + 8) && *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 16) )
    return *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 24);
  return result;
}
