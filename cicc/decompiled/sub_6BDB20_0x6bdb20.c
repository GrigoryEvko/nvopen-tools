// Function: sub_6BDB20
// Address: 0x6bdb20
//
__int64 __fastcall sub_6BDB20(unsigned int a1, __int64 a2)
{
  _QWORD *v4; // rdi
  __int64 v5; // r13
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE v12[192]; // [rsp+10h] [rbp-C0h] BYREF

  v11 = 0;
  sub_6E2250(v12, &v10, 4, a1, a2, 0);
  if ( qword_4D03C50 && (v4 = *(_QWORD **)(qword_4D03C50 + 136LL)) != 0 && *v4 )
  {
    v5 = sub_6E1C80(v4);
  }
  else
  {
    v6 = 0;
    if ( a2 && (*(_BYTE *)(a2 + 179) & 1) != 0 )
      v6 = &v11;
    v5 = sub_6BA760(a1, (__int64)v6);
  }
  sub_6E2C70(v10, a1, a2, 0);
  if ( *(_BYTE *)(v5 + 8) == 3 )
  {
    *(_BYTE *)(a2 + 179) |= 2u;
    v7 = v11;
    v8 = qword_4F06BC0;
    *(_QWORD *)(v11 + 24) = a2;
    v5 = *(_QWORD *)(v7 + 8);
    *(_QWORD *)(v7 + 32) = v8;
  }
  return v5;
}
