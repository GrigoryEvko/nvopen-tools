// Function: sub_D65C20
// Address: 0xd65c20
//
__int64 __fastcall sub_D65C20(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r12
  _BYTE v14[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v3 = sub_D63080(a1, *(unsigned __int8 **)(a2 - 64));
  v5 = v4;
  v7 = sub_D63080(a1, *(unsigned __int8 **)(a2 - 32));
  v8 = v6;
  if ( v7 == 0 || v5 == 0 || v3 == 0 || !v6 )
    return 0;
  if ( v7 == v3 && v6 == v5 )
    return v3;
  v10 = *(_QWORD *)(a2 - 96);
  v15 = 257;
  v11 = sub_B36550((unsigned int **)(a1 + 24), v10, v3, v7, (__int64)v14, 0);
  v12 = *(_QWORD *)(a2 - 96);
  v13 = v11;
  v15 = 257;
  sub_B36550((unsigned int **)(a1 + 24), v12, v5, v8, (__int64)v14, 0);
  return v13;
}
