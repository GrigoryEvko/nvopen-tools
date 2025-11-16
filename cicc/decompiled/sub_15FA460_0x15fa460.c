// Function: sub_15FA460
// Address: 0x15fa460
//
bool __fastcall sub_15FA460(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
    return *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 11;
  return result;
}
