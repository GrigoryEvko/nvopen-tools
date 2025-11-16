// Function: sub_B2BCD0
// Address: 0xb2bcd0
//
__int64 __fastcall sub_B2BCD0(__int64 a1)
{
  int v1; // esi
  __int64 v2; // rax
  __int64 v4; // [rsp+8h] [rbp-8h] BYREF

  v1 = *(_DWORD *)(a1 + 32);
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 120LL);
  v2 = sub_A744E0(&v4, v1);
  return sub_B2B600(v2);
}
