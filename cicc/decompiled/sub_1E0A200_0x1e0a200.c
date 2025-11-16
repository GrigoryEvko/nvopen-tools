// Function: sub_1E0A200
// Address: 0x1e0a200
//
__int64 __fastcall sub_1E0A200(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax

  sub_1DD5C30(a2);
  result = *(_QWORD *)(a1 + 312);
  *a2 = result;
  *(_QWORD *)(a1 + 312) = a2;
  return result;
}
