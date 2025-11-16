// Function: sub_F519F0
// Address: 0xf519f0
//
void __fastcall sub_F519F0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v10; // [rsp+8h] [rbp-48h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (1 - v4)) + 24LL);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (2 - v4)) + 24LL);
  sub_AE7A80((__int64)v11, a1);
  if ( sub_AF4770(v5) || !sub_AF4730(v5) && (unsigned __int8)sub_F506A0(*(_QWORD *)(v3 + 8), a1) )
  {
    sub_F4EE60(a3, v3, v10, v5, (__int64)v11, v6, a2 + 24, 0);
  }
  else
  {
    v7 = sub_ACADE0(*(__int64 ***)(v3 + 8));
    sub_F4EE60(a3, v7, v10, v5, (__int64)v11, v8, a2 + 24, 0);
  }
  if ( v11[0] )
    sub_B91220((__int64)v11, v11[0]);
}
