// Function: sub_16015D0
// Address: 0x16015d0
//
__int64 __fastcall sub_16015D0(__int64 a1)
{
  _QWORD *v1; // r13
  __int64 v2; // r14
  _QWORD *v3; // r15
  __int64 v4; // rax
  __int64 v5; // r12
  char v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+10h] [rbp-30h]

  v1 = *(_QWORD **)(a1 - 72);
  v2 = *(_QWORD *)(a1 - 48);
  v8 = 257;
  v3 = *(_QWORD **)(a1 - 24);
  v4 = sub_1648A60(56, 3);
  v5 = v4;
  if ( v4 )
    sub_15FA660(v4, v1, v2, v3, (__int64)v7, 0);
  return v5;
}
