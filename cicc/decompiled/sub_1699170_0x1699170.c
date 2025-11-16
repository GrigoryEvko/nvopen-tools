// Function: sub_1699170
// Address: 0x1699170
//
__int64 __fastcall sub_1699170(__int64 a1, __int64 a2, __int64 a3)
{
  sub_1698320((_QWORD *)a1, a2);
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF0 | 2;
  sub_1698870(a1);
  *(_WORD *)(a1 + 16) = *(_DWORD *)(a2 + 4) - 1;
  *(_QWORD *)sub_1698470(a1) = a3;
  return sub_1698EC0((__int16 **)a1, 0, 0);
}
