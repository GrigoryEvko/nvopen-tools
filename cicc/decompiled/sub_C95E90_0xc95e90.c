// Function: sub_C95E90
// Address: 0xc95e90
//
__int64 __fastcall sub_C95E90(__int64 a1)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int16 v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-48h]

  v3 = **(_QWORD **)(a1 + 8);
  v2 = *(_QWORD *)a1;
  v9 = 3;
  v10 = v3;
  sub_C6B410(v2, (unsigned __int8 *)"count", 5u);
  sub_C6C710(v2, &v9, v4);
  sub_C6AE10(v2);
  sub_C6BC50(&v9);
  v5 = *(_QWORD *)a1;
  v6 = **(_QWORD **)(a1 + 16) / **(_QWORD **)(a1 + 8);
  v9 = 3;
  v10 = v6 / 0x3E8;
  sub_C6B410(v5, "avg ms", 6u);
  sub_C6C710(v5, &v9, v7);
  sub_C6AE10(v5);
  return sub_C6BC50(&v9);
}
