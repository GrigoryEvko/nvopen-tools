// Function: sub_3400E40
// Address: 0x3400e40
//
unsigned __int8 *__fastcall sub_3400E40(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v13; // [rsp+8h] [rbp-38h]

  v13 = *(_QWORD *)(a1 + 16);
  v9 = sub_2E79000(*(__int64 **)(a1 + 40));
  v10 = sub_2FE6750(v13, a3, a4, v9);
  return sub_3400BD0(a1, a2, a5, v10, v11, 0, a6, 0);
}
