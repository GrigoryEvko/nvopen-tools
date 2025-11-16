// Function: sub_172AC10
// Address: 0x172ac10
//
unsigned __int8 *__fastcall sub_172AC10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        double a5,
        double a6,
        double a7)
{
  unsigned __int8 *v9; // r12
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 *v15; // r13
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  bool v19; // zf
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  unsigned __int8 *v23; // [rsp+8h] [rbp-48h] BYREF
  char v24[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v25; // [rsp+20h] [rbp-30h]

  v9 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)(a3 + 16) <= 0x10u )
  {
    if ( sub_1593BB0(a3, a2, a3, (__int64)a4) )
      return v9;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v9 = (unsigned __int8 *)sub_15A2D10((__int64 *)a2, a3, a5, a6, a7);
      v11 = sub_14DBA30((__int64)v9, *(_QWORD *)(a1 + 96), 0);
      if ( v11 )
        return (unsigned __int8 *)v11;
      return v9;
    }
  }
  v25 = 257;
  v13 = sub_15FB440(27, (__int64 *)a2, a3, (__int64)v24, 0);
  v14 = *(_QWORD *)(a1 + 8);
  v9 = (unsigned __int8 *)v13;
  if ( v14 )
  {
    v15 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v14 + 40, v13);
    v16 = *((_QWORD *)v9 + 3);
    v17 = *v15;
    *((_QWORD *)v9 + 4) = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v9 + 3) = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v9 + 24;
    *v15 = *v15 & 7 | (unsigned __int64)(v9 + 24);
  }
  sub_164B780((__int64)v9, a4);
  v19 = *(_QWORD *)(a1 + 80) == 0;
  v23 = v9;
  if ( v19 )
    sub_4263D6(v9, a4, v18);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v23);
  v20 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return v9;
  v23 = *(unsigned __int8 **)a1;
  sub_1623A60((__int64)&v23, v20, 2);
  v21 = *((_QWORD *)v9 + 6);
  if ( v21 )
    sub_161E7C0((__int64)(v9 + 48), v21);
  v22 = v23;
  *((_QWORD *)v9 + 6) = v23;
  if ( !v22 )
    return v9;
  sub_1623210((__int64)&v23, v22, (__int64)(v9 + 48));
  return v9;
}
