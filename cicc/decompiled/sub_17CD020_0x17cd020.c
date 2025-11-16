// Function: sub_17CD020
// Address: 0x17cd020
//
_QWORD *__fastcall sub_17CD020(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  _QWORD *v9; // r12
  unsigned __int64 *v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  unsigned __int8 *v17; // [rsp+8h] [rbp-48h] BYREF
  char v18[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v19; // [rsp+20h] [rbp-30h]

  v19 = 257;
  v7 = sub_15FDBD0(a2, a3, a4, (__int64)v18, 0);
  v8 = a1[1];
  v9 = (_QWORD *)v7;
  if ( v8 )
  {
    v10 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v8 + 40, v7);
    v11 = v9[3];
    v12 = *v10;
    v9[4] = v10;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v12 | v11 & 7;
    *(_QWORD *)(v12 + 8) = v9 + 3;
    *v10 = *v10 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780((__int64)v9, a5);
  v13 = *a1;
  if ( *a1 )
  {
    v17 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v17, v13, 2);
    v14 = v9[6];
    if ( v14 )
      sub_161E7C0((__int64)(v9 + 6), v14);
    v15 = v17;
    v9[6] = v17;
    if ( v15 )
      sub_1623210((__int64)&v17, v15, (__int64)(v9 + 6));
  }
  return v9;
}
