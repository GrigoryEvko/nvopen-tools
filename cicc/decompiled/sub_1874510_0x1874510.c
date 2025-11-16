// Function: sub_1874510
// Address: 0x1874510
//
__int64 __fastcall sub_1874510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r9
  __int64 v7; // rdi
  __int64 v8; // rdx
  char v9; // al
  __int64 result; // rax
  _QWORD v11[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v13; // [rsp+20h] [rbp-50h]
  _QWORD v14[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v15; // [rsp+40h] [rbp-30h]
  _QWORD v16[2]; // [rsp+50h] [rbp-20h] BYREF
  __int16 v17; // [rsp+60h] [rbp-10h]

  v4 = *(__int64 **)a1;
  v11[1] = a3;
  v5 = *(_QWORD *)(a1 + 8);
  v11[0] = a2;
  v6 = *v4;
  v12[1] = v5;
  v14[0] = v12;
  v13 = 1283;
  v14[1] = "_";
  v16[0] = v14;
  v17 = 1282;
  v7 = v4[6];
  v12[0] = "__typeid_";
  v15 = 770;
  v16[1] = v11;
  v8 = sub_15E57E0(v7, 0, 0, (__int64)v16, a4, v6);
  v9 = *(_BYTE *)(v8 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v8 + 32) = v9;
  result = v9 & 0xF;
  if ( (_BYTE)result != 9 )
    *(_BYTE *)(v8 + 33) |= 0x40u;
  return result;
}
