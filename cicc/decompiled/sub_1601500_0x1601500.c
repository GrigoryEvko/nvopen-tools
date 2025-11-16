// Function: sub_1601500
// Address: 0x1601500
//
__int64 __fastcall sub_1601500(__int64 a1)
{
  __int64 v1; // r14
  _QWORD *v2; // r13
  __int64 v3; // rax
  __int64 v4; // r12
  char v6[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v7; // [rsp+10h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(_QWORD **)(a1 - 48);
  v7 = 257;
  v3 = sub_1648A60(56, 2);
  v4 = v3;
  if ( v3 )
    sub_15FA320(v3, v2, v1, (__int64)v6, 0);
  return v4;
}
