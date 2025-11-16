// Function: sub_1287F30
// Address: 0x1287f30
//
__int64 __fastcall sub_1287F30(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // cc
  __int64 *v5; // rbx
  _QWORD *v6; // r12
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 *v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-68h] BYREF
  char *v16; // [rsp+10h] [rbp-60h] BYREF
  char v17; // [rsp+20h] [rbp-50h]
  char v18; // [rsp+21h] [rbp-4Fh]
  char v19[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v20; // [rsp+40h] [rbp-30h]

  v4 = *(_BYTE *)(a3 + 16) <= 0x10u;
  v5 = *(__int64 **)(a1 + 8);
  v18 = 1;
  v16 = "or";
  v17 = 3;
  if ( v4 )
  {
    v6 = (_QWORD *)a2;
    if ( (unsigned __int8)sub_1593BB0(a3) )
      return (__int64)v6;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      return sub_15A2D10(a2, a3);
  }
  v20 = 257;
  v8 = sub_15FB440(27, a2, a3, v19, 0);
  v9 = v5[1];
  v6 = (_QWORD *)v8;
  if ( v9 )
  {
    v10 = (unsigned __int64 *)v5[2];
    sub_157E9D0(v9 + 40, v8);
    v11 = v6[3];
    v12 = *v10;
    v6[4] = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    v6[3] = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v6 + 3;
    *v10 = *v10 & 7 | (unsigned __int64)(v6 + 3);
  }
  sub_164B780(v6, &v16);
  v13 = *v5;
  if ( !*v5 )
    return (__int64)v6;
  v15 = *v5;
  sub_1623A60(&v15, v13, 2);
  if ( v6[6] )
    sub_161E7C0(v6 + 6);
  v14 = v15;
  v6[6] = v15;
  if ( !v14 )
    return (__int64)v6;
  sub_1623210(&v15, v14, v6 + 6);
  return (__int64)v6;
}
