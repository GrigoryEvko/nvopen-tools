// Function: sub_F51BB0
// Address: 0xf51bb0
//
void __fastcall sub_F51BB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (1 - v4)) + 24LL);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (2 - v4)) + 24LL);
  if ( (unsigned __int8)sub_F506A0(*(_QWORD *)(a2 + 8), a1) )
  {
    sub_AE7A80((__int64)v8, a1);
    sub_F4EE60(a3, a2, v5, v6, (__int64)v8, v7, *(_QWORD *)(a2 + 32), 1);
    if ( v8[0] )
      sub_B91220((__int64)v8, v8[0]);
  }
}
