// Function: sub_B55B40
// Address: 0xb55b40
//
__int64 __fastcall sub_B55B40(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // r9
  __int64 v5; // r12
  __int64 v7; // [rsp+8h] [rbp-68h]
  char v8[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v9; // [rsp+30h] [rbp-40h]

  v1 = *(_QWORD *)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 96);
  v9 = 257;
  v7 = *(_QWORD *)(a1 - 32);
  v3 = sub_BD2C40(72, 3);
  v5 = v3;
  if ( v3 )
    sub_B4DFA0(v3, v2, v1, v7, (__int64)v8, v4, 0, 0);
  return v5;
}
