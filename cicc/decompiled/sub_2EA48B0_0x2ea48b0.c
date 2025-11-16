// Function: sub_2EA48B0
// Address: 0x2ea48b0
//
__int64 __fastcall sub_2EA48B0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdi
  __int64 v4; // [rsp+0h] [rbp-10h] BYREF
  __int64 *v5; // [rsp+8h] [rbp-8h] BYREF

  v1 = *(_QWORD *)(a1 + 40);
  v4 = a1;
  v2 = *(_QWORD *)(a1 + 32);
  v5 = &v4;
  return sub_2EA4160(v2, v1, &v5, 0);
}
