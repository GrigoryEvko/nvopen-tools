// Function: sub_2623820
// Address: 0x2623820
//
__int64 __fastcall sub_2623820(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  __int64 v10; // rdx
  char v11; // al
  __int64 result; // rax
  _QWORD v13[4]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v14; // [rsp+20h] [rbp-70h]
  _QWORD v15[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v16; // [rsp+50h] [rbp-40h]
  _QWORD v17[4]; // [rsp+60h] [rbp-30h] BYREF
  __int16 v18; // [rsp+80h] [rbp-10h]

  v5 = *(__int64 **)a1;
  v6 = *(_QWORD **)(a1 + 8);
  v7 = **(_QWORD **)a1;
  v14 = 1283;
  v13[0] = "__typeid_";
  v13[2] = *v6;
  v8 = v6[1];
  v16 = 770;
  v9 = (_QWORD *)v5[8];
  v13[3] = v8;
  v15[0] = v13;
  v15[2] = "_";
  v17[0] = v15;
  v17[2] = a2;
  v17[3] = a3;
  v18 = 1282;
  v10 = sub_B30500(v9, 0, 0, (__int64)v17, a4, v7);
  v11 = *(_BYTE *)(v10 + 32) & 0xCF | 0x10;
  *(_BYTE *)(v10 + 32) = v11;
  result = v11 & 0xF;
  if ( (_BYTE)result != 9 )
    *(_BYTE *)(v10 + 33) |= 0x40u;
  return result;
}
