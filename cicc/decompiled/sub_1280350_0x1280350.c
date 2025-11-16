// Function: sub_1280350
// Address: 0x1280350
//
__int64 __fastcall sub_1280350(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_12A2A10();
  if ( !result )
    sub_127B550("could not lookup variable in map!", (_DWORD *)(a2 + 64), 1);
  return result;
}
