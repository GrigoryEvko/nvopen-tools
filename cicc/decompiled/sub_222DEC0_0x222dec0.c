// Function: sub_222DEC0
// Address: 0x222dec0
//
_BOOL8 __fastcall sub_222DEC0(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax

  sub_22550C0();
  sub_222DE10((_QWORD *)a1, a1 + 208);
  *(_QWORD *)(a1 + 232) = a2;
  result = a2 == 0;
  *(_DWORD *)(a1 + 224) = 0;
  *(_BYTE *)(a1 + 228) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 28) = 0;
  *(_DWORD *)(a1 + 32) = result;
  return result;
}
