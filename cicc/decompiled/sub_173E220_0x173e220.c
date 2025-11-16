// Function: sub_173E220
// Address: 0x173e220
//
_QWORD *__fastcall sub_173E220(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  __int64 *v8; // r9
  __int64 v9; // r10
  __int64 v13; // r15
  _QWORD *v14; // r12
  __int64 v16; // rax
  int v17; // r8d
  __int64 v18; // rdi
  unsigned __int64 *v19; // r13
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  bool v23; // zf
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  int v28; // [rsp+8h] [rbp-58h]
  _QWORD v29[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v30; // [rsp+20h] [rbp-40h]

  v8 = (__int64 *)a2;
  v9 = a3;
  if ( *(_BYTE *)(a2 + 16) > 0x10u
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v13 = sub_15A2A30((__int64 *)0xC, (__int64 *)a2, a3, 0, 0, a6, a7, a8),
        (v14 = (_QWORD *)sub_14DBA30(v13, *(_QWORD *)(a1 + 96), 0)) == 0)
    && (v8 = (__int64 *)a2, v9 = a3, (v14 = (_QWORD *)v13) == 0) )
  {
    v30 = 257;
    v16 = sub_15FB440(12, v8, v9, (__int64)v29, 0);
    v17 = *(_DWORD *)(a1 + 40);
    v14 = (_QWORD *)v16;
    if ( a5 || (a5 = *(_QWORD *)(a1 + 32)) != 0 )
    {
      v28 = *(_DWORD *)(a1 + 40);
      sub_1625C10(v16, 3, a5);
      v17 = v28;
    }
    sub_15F2440((__int64)v14, v17);
    v18 = *(_QWORD *)(a1 + 8);
    if ( v18 )
    {
      v19 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v18 + 40, (__int64)v14);
      v20 = v14[3];
      v21 = *v19;
      v14[4] = v19;
      v21 &= 0xFFFFFFFFFFFFFFF8LL;
      v14[3] = v21 | v20 & 7;
      *(_QWORD *)(v21 + 8) = v14 + 3;
      *v19 = *v19 & 7 | (unsigned __int64)(v14 + 3);
    }
    sub_164B780((__int64)v14, a4);
    v23 = *(_QWORD *)(a1 + 80) == 0;
    v29[0] = v14;
    if ( v23 )
      sub_4263D6(v14, a4, v22);
    (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v29);
    v24 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v29[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v29, v24, 2);
      v25 = v14[6];
      if ( v25 )
        sub_161E7C0((__int64)(v14 + 6), v25);
      v26 = (unsigned __int8 *)v29[0];
      v14[6] = v29[0];
      if ( v26 )
        sub_1623210((__int64)v29, v26, (__int64)(v14 + 6));
    }
  }
  return v14;
}
