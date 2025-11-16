// Function: sub_B34CE0
// Address: 0xb34ce0
//
__int64 __fastcall sub_B34CE0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v14[4]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v15[32]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v16; // [rsp+50h] [rbp-30h]

  v7 = (unsigned int)(1LL << a4);
  v8 = *(_QWORD *)(a2 + 8);
  v9 = *(_QWORD *)(a3 + 8);
  v14[0] = a2;
  v14[1] = a3;
  v13[0] = v8;
  v10 = *(_QWORD *)(a1 + 72);
  v13[1] = v9;
  v11 = sub_BCB2D0(v10);
  v14[2] = sub_ACD640(v11, v7, 0);
  v16 = 257;
  v14[3] = a5;
  return sub_B34BE0(a1, 0xE6u, (int)v14, 4, (__int64)v13, 2, (__int64)v15);
}
