// Function: sub_E653E0
// Address: 0xe653e0
//
_BOOL8 __fastcall sub_E653E0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  return a3 > 0xA
      && (*(_QWORD *)a2 == 0x2E617461646F722ELL && *(_WORD *)(a2 + 8) == 29811 && *(_BYTE *)(a2 + 10) == 114
       || *(_QWORD *)a2 == 0x2E617461646F722ELL && *(_WORD *)(a2 + 8) == 29539 && *(_BYTE *)(a2 + 10) == 116);
}
