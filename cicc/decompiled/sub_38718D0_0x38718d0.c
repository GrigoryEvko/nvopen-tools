// Function: sub_38718D0
// Address: 0x38718d0
//
_QWORD *__fastcall sub_38718D0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        char a6,
        double a7,
        double a8,
        double a9)
{
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  unsigned __int64 *v23; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-58h] BYREF
  char v25[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v26 = 257;
    v16 = sub_15FB440(13, (__int64 *)a2, a3, (__int64)v25, 0);
    v17 = a1[1];
    v13 = (_QWORD *)v16;
    if ( v17 )
    {
      v23 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v17 + 40, v16);
      v18 = *v23;
      v19 = v13[3] & 7LL;
      v13[4] = v23;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      v13[3] = v18 | v19;
      *(_QWORD *)(v18 + 8) = v13 + 3;
      *v23 = *v23 & 7 | (unsigned __int64)(v13 + 3);
    }
    sub_164B780((__int64)v13, a4);
    v20 = *a1;
    if ( *a1 )
    {
      v24 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v24, v20, 2);
      v21 = v13[6];
      if ( v21 )
        sub_161E7C0((__int64)(v13 + 6), v21);
      v22 = v24;
      v13[6] = v24;
      if ( v22 )
        sub_1623210((__int64)&v24, v22, (__int64)(v13 + 6));
    }
    if ( a5 )
      sub_15F2310((__int64)v13, 1);
    if ( a6 )
      sub_15F2330((__int64)v13, 1);
  }
  else
  {
    v13 = (_QWORD *)sub_15A2B60((__int64 *)a2, a3, a5, a6, a7, a8, a9);
    v14 = sub_14DBA30((__int64)v13, a1[8], 0);
    if ( v14 )
      return (_QWORD *)v14;
  }
  return v13;
}
