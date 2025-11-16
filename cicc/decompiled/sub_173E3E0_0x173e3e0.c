// Function: sub_173E3E0
// Address: 0x173e3e0
//
unsigned __int8 *__fastcall sub_173E3E0(
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
  __int64 v13; // rax
  unsigned __int8 *v14; // r12
  __int64 v15; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  bool v22; // zf
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  unsigned __int64 *v26; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-58h] BYREF
  char v28[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v29; // [rsp+30h] [rbp-40h]

  v13 = sub_15A0680(*(_QWORD *)a2, a3, 0);
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v13 + 16) > 0x10u )
  {
    v29 = 257;
    v17 = sub_15FB440(23, (__int64 *)a2, v13, (__int64)v28, 0);
    v18 = *(_QWORD *)(a1 + 8);
    v14 = (unsigned __int8 *)v17;
    if ( v18 )
    {
      v26 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v18 + 40, v17);
      v19 = *v26;
      v20 = *((_QWORD *)v14 + 3) & 7LL;
      *((_QWORD *)v14 + 4) = v26;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v14 + 3) = v19 | v20;
      *(_QWORD *)(v19 + 8) = v14 + 24;
      *v26 = *v26 & 7 | (unsigned __int64)(v14 + 24);
    }
    sub_164B780((__int64)v14, a4);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v27 = v14;
    if ( v22 )
      sub_4263D6(v14, a4, v21);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v27);
    v23 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v27 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v27, v23, 2);
      v24 = *((_QWORD *)v14 + 6);
      if ( v24 )
        sub_161E7C0((__int64)(v14 + 48), v24);
      v25 = v27;
      *((_QWORD *)v14 + 6) = v27;
      if ( v25 )
        sub_1623210((__int64)&v27, v25, (__int64)(v14 + 48));
    }
    if ( a5 )
      sub_15F2310((__int64)v14, 1);
    if ( a6 )
      sub_15F2330((__int64)v14, 1);
  }
  else
  {
    v14 = (unsigned __int8 *)sub_15A2D50((__int64 *)a2, v13, a5, a6, a7, a8, a9);
    v15 = sub_14DBA30((__int64)v14, *(_QWORD *)(a1 + 96), 0);
    if ( v15 )
      return (unsigned __int8 *)v15;
  }
  return v14;
}
