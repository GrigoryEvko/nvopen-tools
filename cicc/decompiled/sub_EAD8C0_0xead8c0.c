// Function: sub_EAD8C0
// Address: 0xead8c0
//
char __fastcall sub_EAD8C0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  char result; // al
  _QWORD v5[4]; // [rsp+0h] [rbp-50h] BYREF
  char v6; // [rsp+20h] [rbp-30h]
  char v7; // [rsp+21h] [rbp-2Fh]

  v5[0] = 0;
  result = sub_EAC4D0(a1, a2, (__int64)v5);
  if ( !result )
  {
    *a3 = sub_ECD6B0(*(_QWORD *)(a1 + 48));
    v7 = 1;
    v5[0] = "expected ')'";
    v6 = 3;
    return sub_ECE210(a1, 18, v5);
  }
  return result;
}
