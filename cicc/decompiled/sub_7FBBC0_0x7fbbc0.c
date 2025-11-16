// Function: sub_7FBBC0
// Address: 0x7fbbc0
//
__int64 __fastcall sub_7FBBC0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v4; // [rsp-10h] [rbp-40h]
  __m128i v5[3]; // [rsp+0h] [rbp-30h] BYREF

  sub_7E1780(a2, (__int64)v5);
  sub_72A420((__int64 *)a1);
  v2 = (__int64 *)sub_73E230(a1, (__int64)v5);
  sub_7FB7C0(*(_QWORD *)(a1 + 120), 1u, v2, 0, 0, 0, v5);
  *(_BYTE *)(a1 + 177) = 0;
  return v4;
}
