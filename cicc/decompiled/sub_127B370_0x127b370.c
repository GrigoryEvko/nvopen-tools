// Function: sub_127B370
// Address: 0x127b370
//
__int64 __fastcall sub_127B370(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 136);
  if ( !result )
    return *(_QWORD *)(a1 + 8);
  return result;
}
