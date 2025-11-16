// Function: sub_1875AC0
// Address: 0x1875ac0
//
__int64 __fastcall sub_1875AC0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 *v6; // rax
  __int64 v7; // r12
  char v8; // al
  _QWORD v10[2]; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int16 v12; // [rsp+20h] [rbp-90h]
  _QWORD v13[2]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v14; // [rsp+40h] [rbp-70h]
  _QWORD v15[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v16; // [rsp+60h] [rbp-50h]
  __int64 v17[2]; // [rsp+70h] [rbp-40h] BYREF
  __int64 v18; // [rsp+80h] [rbp-30h] BYREF

  v3 = *a1;
  v10[0] = a2;
  v10[1] = a3;
  v4 = *v3;
  v5 = v3[8];
  v16 = 1282;
  v6 = a1[1];
  v11[0] = "__typeid_";
  v12 = 1283;
  v11[1] = v6;
  v13[0] = v11;
  v13[1] = "_";
  v15[0] = v13;
  v14 = 770;
  v15[1] = v10;
  sub_16E2FC0(v17, (__int64)v15);
  v7 = sub_1632210(v4, v17[0], v17[1], v5);
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0], v18 + 1);
  if ( *(_BYTE *)(v7 + 16) == 3 )
  {
    v8 = *(_BYTE *)(v7 + 32) & 0xCF | 0x10;
    *(_BYTE *)(v7 + 32) = v8;
    if ( (v8 & 0xF) != 9 )
      *(_BYTE *)(v7 + 33) |= 0x40u;
  }
  return sub_15A4510((__int64 ***)v7, (__int64 **)(*a1)[7], 0);
}
