// Function: sub_14C50F0
// Address: 0x14c50f0
//
__int64 __fastcall sub_14C50F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  char v19[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    return sub_15A3950(a2, a3, a4, 0);
  v20 = 257;
  v9 = sub_1648A60(56, 3);
  v10 = (_QWORD *)v9;
  if ( v9 )
    sub_15FA660(v9, a2, a3, a4, v19, 0);
  v11 = a1[1];
  if ( v11 )
  {
    v12 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v11 + 40, v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, a5);
  v15 = *a1;
  if ( *a1 )
  {
    v18 = *a1;
    sub_1623A60(&v18, v15, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v16 = v18;
    v10[6] = v18;
    if ( v16 )
      sub_1623210(&v18, v16, v10 + 6);
  }
  return (__int64)v10;
}
