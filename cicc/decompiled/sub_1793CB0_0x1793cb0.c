// Function: sub_1793CB0
// Address: 0x1793cb0
//
unsigned __int8 *__fastcall sub_1793CB0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 *a4,
        double a5,
        double a6,
        double a7)
{
  unsigned __int8 *v8; // r12
  __int64 v10; // r13
  unsigned __int8 v11; // al
  __int64 v12; // rax
  unsigned int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  bool v21; // zf
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  unsigned __int8 *v25; // [rsp+8h] [rbp-58h] BYREF
  char v26[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v27; // [rsp+20h] [rbp-40h]

  v8 = a2;
  v10 = sub_15A1070(*(_QWORD *)a2, a3);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 > 0x10u )
    goto LABEL_10;
  if ( v11 == 13 )
  {
    v14 = *(_DWORD *)(v10 + 32);
    if ( v14 > 0x40 )
    {
      if ( v14 == (unsigned int)sub_16A58F0(v10 + 24) )
        return v8;
      if ( a2[16] <= 0x10u )
        goto LABEL_4;
      goto LABEL_10;
    }
    if ( *(_QWORD *)(v10 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) )
      return v8;
  }
  if ( a2[16] <= 0x10u )
  {
LABEL_4:
    v8 = (unsigned __int8 *)sub_15A2CF0((__int64 *)a2, v10, a5, a6, a7);
    v12 = sub_14DBA30((__int64)v8, *(_QWORD *)(a1 + 96), 0);
    if ( v12 )
      return (unsigned __int8 *)v12;
    return v8;
  }
LABEL_10:
  v27 = 257;
  v15 = sub_15FB440(26, (__int64 *)a2, v10, (__int64)v26, 0);
  v16 = *(_QWORD *)(a1 + 8);
  v8 = (unsigned __int8 *)v15;
  if ( v16 )
  {
    v17 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v16 + 40, v15);
    v18 = *((_QWORD *)v8 + 3);
    v19 = *v17;
    *((_QWORD *)v8 + 4) = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v8 + 3) = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v8 + 24;
    *v17 = *v17 & 7 | (unsigned __int64)(v8 + 24);
  }
  sub_164B780((__int64)v8, a4);
  v21 = *(_QWORD *)(a1 + 80) == 0;
  v25 = v8;
  if ( v21 )
    sub_4263D6(v8, a4, v20);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v25);
  v22 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v25 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v25, v22, 2);
    v23 = *((_QWORD *)v8 + 6);
    if ( v23 )
      sub_161E7C0((__int64)(v8 + 48), v23);
    v24 = v25;
    *((_QWORD *)v8 + 6) = v25;
    if ( v24 )
      sub_1623210((__int64)&v25, v24, (__int64)(v8 + 48));
  }
  return v8;
}
