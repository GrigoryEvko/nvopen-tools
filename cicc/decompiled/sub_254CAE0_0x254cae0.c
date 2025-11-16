// Function: sub_254CAE0
// Address: 0x254cae0
//
__int64 __fastcall sub_254CAE0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_25096F0((_QWORD *)(*(_QWORD *)(a1 + 8) + 72LL));
  **(_BYTE **)a1 &= sub_250C180(a2, v2);
  return sub_252BB70(*(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 8), a2, 1);
}
