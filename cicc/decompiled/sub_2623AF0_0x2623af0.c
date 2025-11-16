// Function: sub_2623AF0
// Address: 0x2623af0
//
_BYTE *__fastcall sub_2623AF0(__int64 *a1, __int64 *a2, void *a3, void *a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  _BYTE *v8; // r12
  char v9; // al
  __int64 v11[2]; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v12; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v13[4]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v14; // [rsp+40h] [rbp-80h]
  _QWORD v15[4]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v16; // [rsp+70h] [rbp-50h]
  void *v17[4]; // [rsp+80h] [rbp-40h] BYREF
  __int16 v18; // [rsp+A0h] [rbp-20h]

  v14 = 1283;
  v4 = *a1;
  v13[0] = "__typeid_";
  v5 = *a2;
  v6 = a1[10];
  v17[2] = a3;
  v13[2] = v5;
  v7 = a2[1];
  v16 = 770;
  v13[3] = v7;
  v15[0] = v13;
  v15[2] = "_";
  v18 = 1282;
  v17[0] = v15;
  v17[3] = a4;
  sub_CA0F50(v11, v17);
  v8 = sub_BA8D60(v4, v11[0], v11[1], v6);
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0]);
  if ( *v8 == 3 )
  {
    v9 = v8[32] & 0xCF | 0x10;
    v8[32] = v9;
    if ( (v9 & 0xF) != 9 )
      v8[33] |= 0x40u;
  }
  return v8;
}
