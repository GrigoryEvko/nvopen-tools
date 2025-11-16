// Function: sub_10C5400
// Address: 0x10c5400
//
bool __fastcall sub_10C5400(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v4; // rax
  __int64 v5; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 58 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( v4 )
  {
    **(_QWORD **)a1 = v4;
    result = 1;
    v5 = *(_QWORD *)(a2 - 32);
    if ( v5 == *(_QWORD *)(a1 + 8) )
      return result;
  }
  else
  {
    v5 = *(_QWORD *)(a2 - 32);
  }
  if ( !v5 )
    return 0;
  **(_QWORD **)a1 = v5;
  return *(_QWORD *)(a2 - 64) == *(_QWORD *)(a1 + 8);
}
