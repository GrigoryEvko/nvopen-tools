// Function: sub_171CBD0
// Address: 0x171cbd0
//
unsigned __int8 *__fastcall sub_171CBD0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        double a8)
{
  char v9; // r13
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  bool v20; // zf
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  unsigned __int64 *v24; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v25; // [rsp+18h] [rbp-58h] BYREF
  char v26[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v27; // [rsp+30h] [rbp-40h]

  v9 = a4;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v27 = 257;
    v15 = sub_15FB530((__int64 *)a2, (__int64)v26, 0, a4);
    v16 = *(_QWORD *)(a1 + 8);
    v11 = (unsigned __int8 *)v15;
    if ( v16 )
    {
      v24 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v16 + 40, v15);
      v17 = *v24;
      v18 = *((_QWORD *)v11 + 3) & 7LL;
      *((_QWORD *)v11 + 4) = v24;
      v17 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v11 + 3) = v17 | v18;
      *(_QWORD *)(v17 + 8) = v11 + 24;
      *v24 = *v24 & 7 | (unsigned __int64)(v11 + 24);
    }
    sub_164B780((__int64)v11, a3);
    v20 = *(_QWORD *)(a1 + 80) == 0;
    v25 = v11;
    if ( v20 )
      sub_4263D6(v11, a3, v19);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v25);
    v21 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v25 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v25, v21, 2);
      v22 = *((_QWORD *)v11 + 6);
      if ( v22 )
        sub_161E7C0((__int64)(v11 + 48), v22);
      v23 = v25;
      *((_QWORD *)v11 + 6) = v25;
      if ( v23 )
        sub_1623210((__int64)&v25, v23, (__int64)(v11 + 48));
    }
    if ( v9 )
      sub_15F2310((__int64)v11, 1);
    if ( a5 )
      sub_15F2330((__int64)v11, 1);
  }
  else
  {
    v11 = (unsigned __int8 *)sub_15A2B90((__int64 *)a2, (unsigned __int8)a4, a5, a4, a6, a7, a8);
    v12 = sub_14DBA30((__int64)v11, *(_QWORD *)(a1 + 96), 0);
    if ( v12 )
      return (unsigned __int8 *)v12;
  }
  return v11;
}
