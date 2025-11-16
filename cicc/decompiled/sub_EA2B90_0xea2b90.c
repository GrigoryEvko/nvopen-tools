// Function: sub_EA2B90
// Address: 0xea2b90
//
__int64 __fastcall sub_EA2B90(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rdi
  _QWORD v8[3]; // [rsp+8h] [rbp-20h] BYREF

  *(_BYTE *)(a1 + 32) = 1;
  v6 = *(__int64 **)(a1 + 248);
  v8[0] = a4;
  v8[1] = a5;
  sub_C91CB0(v6, a2, 0, a3, (__int64)v8, 1, 0, 0, 1u);
  sub_EA2AE0((_QWORD *)a1);
  return 1;
}
