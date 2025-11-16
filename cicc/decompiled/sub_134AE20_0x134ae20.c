// Function: sub_134AE20
// Address: 0x134ae20
//
__int64 __fastcall sub_134AE20(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v6; // r12
  __int64 v7; // r12
  __int64 result; // rax

  *a2 += *(_QWORD *)(a1 + 8);
  v6 = sub_13427E0(a1 + 192);
  *a3 += sub_13427E0(a1 + 9848) + v6;
  v7 = sub_13427E0(a1 + 19632);
  result = sub_13427E0(a1 + 29288);
  *a4 += result + v7;
  return result;
}
