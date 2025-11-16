// Function: sub_1B0E450
// Address: 0x1b0e450
//
_QWORD *__fastcall sub_1B0E450(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  _QWORD *v10; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int8 *v20; // [rsp+8h] [rbp-58h] BYREF
  char v21[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v22; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v10 = (_QWORD *)sub_15A2A30((__int64 *)0x14, (__int64 *)a2, a3, 0, 0, a5, a6, a7)) == 0 )
  {
    v22 = 257;
    v12 = sub_15FB440(20, (__int64 *)a2, a3, (__int64)v21, 0);
    v13 = a1[1];
    v10 = (_QWORD *)v12;
    if ( v13 )
    {
      v14 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v13 + 40, v12);
      v15 = v10[3];
      v16 = *v14;
      v10[4] = v14;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v16 | v15 & 7;
      *(_QWORD *)(v16 + 8) = v10 + 3;
      *v14 = *v14 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780((__int64)v10, a4);
    v17 = *a1;
    if ( *a1 )
    {
      v20 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v20, v17, 2);
      v18 = v10[6];
      if ( v18 )
        sub_161E7C0((__int64)(v10 + 6), v18);
      v19 = v20;
      v10[6] = v20;
      if ( v19 )
        sub_1623210((__int64)&v20, v19, (__int64)(v10 + 6));
    }
  }
  return v10;
}
