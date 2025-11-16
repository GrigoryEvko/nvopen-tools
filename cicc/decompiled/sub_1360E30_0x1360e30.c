// Function: sub_1360E30
// Address: 0x1360e30
//
bool __fastcall sub_1360E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  unsigned __int8 v5; // al

  result = 0;
  if ( a4 == -1 )
    return result;
  result = (*(_BYTE *)(a1 + 17) & 2) != 0;
  if ( (*(_BYTE *)(a1 + 17) & 2) == 0 )
    return result;
  v5 = *(_BYTE *)(*(_QWORD *)a3 + 16LL);
  if ( v5 <= 0x17u )
  {
    if ( v5 == 3 )
      goto LABEL_5;
    return 0;
  }
  if ( v5 != 53 )
    return 0;
LABEL_5:
  result = 0;
  if ( !*(_DWORD *)(a3 + 32) && !*(_DWORD *)(a2 + 32) )
    return *(_QWORD *)(a2 + 8) + *(_QWORD *)(a2 + 16) >= *(_QWORD *)(a3 + 8) + *(_QWORD *)(a3 + 16) + a4;
  return result;
}
