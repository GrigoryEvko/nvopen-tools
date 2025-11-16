// Function: sub_7FCA60
// Address: 0x7fca60
//
__int64 __fastcall sub_7FCA60(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 result; // rax

  v2 = sub_726B30(17);
  v2[9] = a2;
  result = sub_7FCA00(v2);
  *(_QWORD *)(a1 + 184) = a2;
  *(_BYTE *)(a1 + 177) = 2;
  *(_BYTE *)(a2 + 49) |= 2u;
  return result;
}
