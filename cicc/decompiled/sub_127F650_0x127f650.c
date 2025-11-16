// Function: sub_127F650
// Address: 0x127f650
//
__int64 __fastcall sub_127F650(__int64 a1, const __m128i *a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *(_QWORD *)(a1 + 32);
  v5[1] = a1;
  v5[0] = v3;
  v5[2] = a1 + 48;
  v5[3] = *(_QWORD *)(a1 + 40);
  return sub_127D8B0((__int64)v5, a2, a3);
}
