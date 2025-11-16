// Function: sub_C36740
// Address: 0xc36740
//
__int64 __fastcall sub_C36740(__int64 a1, __int64 a2, __int64 a3)
{
  sub_C337F0((_QWORD *)a1, a2);
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF0 | 2;
  sub_C33EE0(a1);
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 8) - 1;
  *(_QWORD *)sub_C33900(a1) = a3;
  return sub_C36450(a1, 1, 0);
}
