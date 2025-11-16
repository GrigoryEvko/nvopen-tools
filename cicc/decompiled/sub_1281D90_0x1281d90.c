// Function: sub_1281D90
// Address: 0x1281d90
//
__int64 __fastcall sub_1281D90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // r12
  unsigned __int64 *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v17[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    return sub_15A2DA0(a2, a3, a5);
  v18 = 257;
  if ( a5 )
  {
    v10 = (_QWORD *)sub_15FB440(25, a2, a3, v17, 0);
    sub_15F2350(v10, 1);
    v9 = a1[1];
    if ( !v9 )
      goto LABEL_7;
    goto LABEL_6;
  }
  v8 = sub_15FB440(25, a2, a3, v17, 0);
  v9 = a1[1];
  v10 = (_QWORD *)v8;
  if ( v9 )
  {
LABEL_6:
    v11 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v9 + 40, v10);
    v12 = v10[3];
    v13 = *v11;
    v10[4] = v11;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v13 | v12 & 7;
    *(_QWORD *)(v13 + 8) = v10 + 3;
    *v11 = *v11 & 7 | (unsigned __int64)(v10 + 3);
  }
LABEL_7:
  sub_164B780(v10, a4);
  v14 = *a1;
  if ( *a1 )
  {
    v16 = *a1;
    sub_1623A60(&v16, v14, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v15 = v16;
    v10[6] = v16;
    if ( v15 )
      sub_1623210(&v16, v15, v10 + 6);
  }
  return (__int64)v10;
}
