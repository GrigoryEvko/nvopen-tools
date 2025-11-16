// Function: sub_D95440
// Address: 0xd95440
//
bool __fastcall sub_D95440(__int64 a1, __int64 a2)
{
  bool result; // al

  result = 0;
  if ( *(_QWORD *)a1 == *(_QWORD *)a2 && *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) )
    return *(_WORD *)(a1 + 16) == *(_WORD *)(a2 + 16);
  return result;
}
