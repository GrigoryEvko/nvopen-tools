// Function: sub_174BAA0
// Address: 0x174baa0
//
_QWORD *__fastcall sub_174BAA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v8; // r10
  __int64 v11; // r15
  _QWORD *v12; // r12
  int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 *v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  bool v22; // zf
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v27; // [rsp+0h] [rbp-60h]
  _QWORD v29[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v30; // [rsp+20h] [rbp-40h]

  v8 = a3;
  if ( *(_BYTE *)(a2 + 16) > 0x10u
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v11 = sub_15A2A30((__int64 *)0x16, (__int64 *)a2, a3, 0, 0, a6, a7, a8),
        (v12 = (_QWORD *)sub_14DBA30(v11, *(_QWORD *)(a1 + 96), 0)) == 0)
    && (v8 = a3, (v12 = (_QWORD *)v11) == 0) )
  {
    v27 = v8;
    v14 = sub_15F24E0(a4);
    v30 = 257;
    v15 = sub_15FB440(22, (__int64 *)a2, v27, (__int64)v29, 0);
    v16 = *(_QWORD *)(a1 + 32);
    v12 = (_QWORD *)v15;
    if ( v16 )
      sub_1625C10(v15, 3, v16);
    sub_15F2440((__int64)v12, v14);
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v18 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, (__int64)v12);
      v19 = v12[3];
      v20 = *v18;
      v12[4] = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v12 + 3;
      *v18 = *v18 & 7 | (unsigned __int64)(v12 + 3);
    }
    sub_164B780((__int64)v12, a5);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v29[0] = v12;
    if ( v22 )
      sub_4263D6(v12, a5, v21);
    (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v29);
    v23 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v29[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v29, v23, 2);
      v24 = v12[6];
      if ( v24 )
        sub_161E7C0((__int64)(v12 + 6), v24);
      v25 = (unsigned __int8 *)v29[0];
      v12[6] = v29[0];
      if ( v25 )
        sub_1623210((__int64)v29, v25, (__int64)(v12 + 6));
    }
  }
  return v12;
}
