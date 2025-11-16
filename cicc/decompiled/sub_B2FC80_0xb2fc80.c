// Function: sub_B2FC80
// Address: 0xb2fc80
//
bool __fastcall sub_B2FC80(__int64 a1)
{
  bool result; // al

  if ( *(_BYTE *)a1 == 3 )
    return (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0;
  result = 0;
  if ( !*(_BYTE *)a1 && a1 + 72 == (*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
    return ((*(_WORD *)(a1 + 34) >> 1) & 0x400) == 0;
  return result;
}
