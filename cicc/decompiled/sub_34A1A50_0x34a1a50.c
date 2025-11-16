// Function: sub_34A1A50
// Address: 0x34a1a50
//
bool __fastcall sub_34A1A50(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( *(_DWORD *)(a1 + 88) == *(_DWORD *)(a2 + 88) && *(_QWORD *)(a1 + 96) == *(_QWORD *)(a2 + 96) )
    return *(_QWORD *)(a1 + 104) == *(_QWORD *)(a2 + 104);
  return result;
}
