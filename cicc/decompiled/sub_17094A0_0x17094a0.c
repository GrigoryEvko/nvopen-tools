// Function: sub_17094A0
// Address: 0x17094a0
//
unsigned __int8 *__fastcall sub_17094A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int8 a5,
        char a6,
        double a7,
        double a8,
        double a9)
{
  unsigned __int8 *v13; // r12
  __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  bool v21; // zf
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  unsigned __int64 *v25; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v26; // [rsp+18h] [rbp-58h] BYREF
  char v27[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v28; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v28 = 257;
    v16 = sub_15FB440(11, (__int64 *)a2, a3, (__int64)v27, 0);
    v17 = *(_QWORD *)(a1 + 8);
    v13 = (unsigned __int8 *)v16;
    if ( v17 )
    {
      v25 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, v16);
      v18 = *v25;
      v19 = *((_QWORD *)v13 + 3) & 7LL;
      *((_QWORD *)v13 + 4) = v25;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v13 + 3) = v18 | v19;
      *(_QWORD *)(v18 + 8) = v13 + 24;
      *v25 = *v25 & 7 | (unsigned __int64)(v13 + 24);
    }
    sub_164B780((__int64)v13, a4);
    v21 = *(_QWORD *)(a1 + 80) == 0;
    v26 = v13;
    if ( v21 )
      sub_4263D6(v13, a4, v20);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v26);
    v22 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v26 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v26, v22, 2);
      v23 = *((_QWORD *)v13 + 6);
      if ( v23 )
        sub_161E7C0((__int64)(v13 + 48), v23);
      v24 = v26;
      *((_QWORD *)v13 + 6) = v26;
      if ( v24 )
        sub_1623210((__int64)&v26, v24, (__int64)(v13 + 48));
    }
    if ( a5 )
      sub_15F2310((__int64)v13, 1);
    if ( a6 )
      sub_15F2330((__int64)v13, 1);
  }
  else
  {
    v13 = (unsigned __int8 *)sub_15A2B30((__int64 *)a2, a3, a5, a6, a7, a8, a9);
    v14 = sub_14DBA30((__int64)v13, *(_QWORD *)(a1 + 96), 0);
    if ( v14 )
      return (unsigned __int8 *)v14;
  }
  return v13;
}
