// Function: sub_942B80
// Address: 0x942b80
//
__int64 __fastcall sub_942B80(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v3; // r12d
  unsigned int v4; // eax

  v2 = sub_941B90(a1, *(_QWORD *)(a2 + 160));
  v3 = -((*(_BYTE *)(a2 + 168) & 2) == 0);
  v4 = sub_91B6E0(a1);
  return sub_ADCAD0(a1 + 16, (v3 & 0xFFFFFFCE) + 66, v2, v4, 0, 0x10000000CLL);
}
