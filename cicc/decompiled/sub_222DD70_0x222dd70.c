// Function: sub_222DD70
// Address: 0x222dd70
//
_BOOL8 __fastcall sub_222DD70(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax

  sub_22550C0();
  sub_222DCC0((_QWORD *)a1, a1 + 208);
  *(_QWORD *)(a1 + 232) = a2;
  *(_WORD *)(a1 + 224) = 0;
  result = a2 == 0;
  *(_DWORD *)(a1 + 28) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 32) = result;
  return result;
}
