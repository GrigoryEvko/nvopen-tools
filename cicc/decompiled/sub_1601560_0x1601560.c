// Function: sub_1601560
// Address: 0x1601560
//
__int64 __fastcall sub_1601560(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r14
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  char v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v8; // [rsp+10h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(_QWORD *)(a1 - 48);
  v8 = 257;
  v3 = *(__int64 **)(a1 - 72);
  v4 = sub_1648A60(56, 3);
  v5 = v4;
  if ( v4 )
    sub_15FA480(v4, v3, v2, v1, (__int64)v7, 0);
  return v5;
}
