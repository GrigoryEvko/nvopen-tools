// Function: sub_2433ED0
// Address: 0x2433ed0
//
__int64 __fastcall sub_2433ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx

  v2 = sub_BAA410(*(_QWORD *)(*(_QWORD *)a1 + 8LL), "hwasan.module_ctor", 0x12u);
  sub_B2F990(a2, v2, v3, v4);
  return sub_2A3ED40(*(_QWORD *)(*(_QWORD *)a1 + 8LL), a2, 0, a2);
}
