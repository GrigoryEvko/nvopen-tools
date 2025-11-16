// Function: sub_72D4C0
// Address: 0x72d4c0
//
__int64 __fastcall sub_72D4C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 result; // rax

  sub_724C70(a2, 6);
  *(_QWORD *)(a2 + 184) = a1;
  *(_BYTE *)(a2 + 176) = 6;
  *(_BYTE *)(a1 + 120) |= 0x10u;
  v2 = (_QWORD *)sub_72CBE0();
  result = sub_72D2E0(v2);
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
