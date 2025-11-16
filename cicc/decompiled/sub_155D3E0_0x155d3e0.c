// Function: sub_155D3E0
// Address: 0x155d3e0
//
bool __fastcall sub_155D3E0(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_BYTE *)(*(_QWORD *)a1 + 16LL) == 2;
  return result;
}
