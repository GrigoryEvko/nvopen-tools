// Function: sub_172B670
// Address: 0x172b670
//
unsigned __int8 *__fastcall sub_172B670(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        double a5,
        double a6,
        double a7)
{
  unsigned __int8 *v10; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rdx
  bool v18; // zf
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v23; // [rsp+18h] [rbp-58h] BYREF
  char v24[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v25; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v22 = sub_15A2A30((__int64 *)0x1C, (__int64 *)a2, a3, 0, 0, a5, a6, a7),
        (v10 = (unsigned __int8 *)sub_14DBA30(v22, *(_QWORD *)(a1 + 96), 0)) == 0)
    && (v10 = (unsigned __int8 *)v22) == 0 )
  {
    v25 = 257;
    v12 = sub_15FB440(28, (__int64 *)a2, a3, (__int64)v24, 0);
    v13 = *(_QWORD *)(a1 + 8);
    v10 = (unsigned __int8 *)v12;
    if ( v13 )
    {
      v14 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v13 + 40, v12);
      v15 = *((_QWORD *)v10 + 3);
      v16 = *v14;
      *((_QWORD *)v10 + 4) = v14;
      v16 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v10 + 3) = v16 | v15 & 7;
      *(_QWORD *)(v16 + 8) = v10 + 24;
      *v14 = *v14 & 7 | (unsigned __int64)(v10 + 24);
    }
    sub_164B780((__int64)v10, a4);
    v18 = *(_QWORD *)(a1 + 80) == 0;
    v23 = v10;
    if ( v18 )
      sub_4263D6(v10, a4, v17);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v23);
    v19 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v23 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v23, v19, 2);
      v20 = *((_QWORD *)v10 + 6);
      if ( v20 )
        sub_161E7C0((__int64)(v10 + 48), v20);
      v21 = v23;
      *((_QWORD *)v10 + 6) = v23;
      if ( v21 )
        sub_1623210((__int64)&v23, v21, (__int64)(v10 + 48));
    }
  }
  return v10;
}
