// Function: sub_156D290
// Address: 0x156d290
//
__int64 __fastcall sub_156D290(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // r12
  unsigned __int64 *v9; // r14
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  char v15[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v16; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    return sub_15A2B00(a2);
  v16 = 257;
  v6 = sub_15FB630(a2, v15, 0);
  v7 = a1[1];
  v8 = (_QWORD *)v6;
  if ( v7 )
  {
    v9 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v7 + 40, v6);
    v10 = v8[3];
    v11 = *v9;
    v8[4] = v9;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    v8[3] = v11 | v10 & 7;
    *(_QWORD *)(v11 + 8) = v8 + 3;
    *v9 = *v9 & 7 | (unsigned __int64)(v8 + 3);
  }
  sub_164B780(v8, a3);
  v12 = *a1;
  if ( *a1 )
  {
    v14 = *a1;
    sub_1623A60(&v14, v12, 2);
    if ( v8[6] )
      sub_161E7C0(v8 + 6);
    v13 = v14;
    v8[6] = v14;
    if ( v13 )
      sub_1623210(&v14, v13, v8 + 6);
  }
  return (__int64)v8;
}
