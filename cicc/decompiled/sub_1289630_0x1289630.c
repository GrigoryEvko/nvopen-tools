// Function: sub_1289630
// Address: 0x1289630
//
__int64 __fastcall sub_1289630(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  _QWORD *v7; // r12
  unsigned __int64 *v8; // r13
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-68h] BYREF
  char v14[16]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+20h] [rbp-50h]
  char v16[16]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v17; // [rsp+40h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A4A70(a2, a3);
  v15 = 257;
  if ( a3 == *(_QWORD *)a2 )
    return a2;
  v17 = 257;
  v5 = sub_15FDFF0(a2, a3, v16, 0);
  v6 = a1[7];
  v7 = (_QWORD *)v5;
  if ( v6 )
  {
    v8 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v6 + 40, v5);
    v9 = v7[3];
    v10 = *v8;
    v7[4] = v8;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    v7[3] = v10 | v9 & 7;
    *(_QWORD *)(v10 + 8) = v7 + 3;
    *v8 = *v8 & 7 | (unsigned __int64)(v7 + 3);
  }
  sub_164B780(v7, v14);
  v11 = a1[6];
  if ( v11 )
  {
    v13 = a1[6];
    sub_1623A60(&v13, v11, 2);
    if ( v7[6] )
      sub_161E7C0(v7 + 6);
    v12 = v13;
    v7[6] = v13;
    if ( v12 )
      sub_1623210(&v13, v12, v7 + 6);
  }
  return (__int64)v7;
}
