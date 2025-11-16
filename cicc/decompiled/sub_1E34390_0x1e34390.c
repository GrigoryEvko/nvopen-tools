// Function: sub_1E34390
// Address: 0x1e34390
//
__int64 __fastcall sub_1E34390(__int64 a1)
{
  return -(*(_QWORD *)(a1 + 8) | ((unsigned int)(1 << *(_WORD *)(a1 + 34)) >> 1))
       & (*(_QWORD *)(a1 + 8) | ((unsigned int)(1 << *(_WORD *)(a1 + 34)) >> 1));
}
