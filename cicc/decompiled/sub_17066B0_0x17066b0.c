// Function: sub_17066B0
// Address: 0x17066b0
//
__int64 __fastcall sub_17066B0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  int v9; // r11d
  __int64 v10; // r10
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  char v19; // al
  int v20; // r12d
  __int64 v21; // rdi
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  bool v26; // zf
  __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  _QWORD v32[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v33; // [rsp+30h] [rbp-40h]

  v9 = a2;
  v10 = a4;
  if ( *(_BYTE *)(a3 + 16) > 0x10u
    || *(_BYTE *)(a4 + 16) > 0x10u
    || (v14 = sub_15A2A30((__int64 *)a2, (__int64 *)a3, a4, 0, 0, a7, a8, a9),
        (v15 = sub_14DBA30(v14, *(_QWORD *)(a1 + 96), 0)) == 0)
    && (v9 = a2, v10 = a4, (v15 = v14) == 0) )
  {
    v33 = 257;
    v17 = sub_15FB440(v9, (__int64 *)a3, v10, (__int64)v32, 0);
    v18 = *(_QWORD *)v17;
    v15 = v17;
    v19 = *(_BYTE *)(*(_QWORD *)v17 + 8LL);
    if ( v19 == 16 )
      v19 = *(_BYTE *)(**(_QWORD **)(v18 + 16) + 8LL);
    if ( (unsigned __int8)(v19 - 1) <= 5u || *(_BYTE *)(v15 + 16) == 76 )
    {
      v20 = *(_DWORD *)(a1 + 40);
      if ( a6 || (a6 = *(_QWORD *)(a1 + 32)) != 0 )
        sub_1625C10(v15, 3, a6);
      sub_15F2440(v15, v20);
    }
    v21 = *(_QWORD *)(a1 + 8);
    if ( v21 )
    {
      v22 = *(__int64 **)(a1 + 16);
      sub_157E9D0(v21 + 40, v15);
      v23 = *(_QWORD *)(v15 + 24);
      v24 = *v22;
      *(_QWORD *)(v15 + 32) = v22;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v15 + 24) = v24 | v23 & 7;
      *(_QWORD *)(v24 + 8) = v15 + 24;
      *v22 = *v22 & 7 | (v15 + 24);
    }
    sub_164B780(v15, a5);
    v26 = *(_QWORD *)(a1 + 80) == 0;
    v32[0] = v15;
    if ( v26 )
      sub_4263D6(v15, a5, v25);
    (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v32);
    v27 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v32[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v32, v27, 2);
      v28 = *(_QWORD *)(v15 + 48);
      if ( v28 )
        sub_161E7C0(v15 + 48, v28);
      v29 = (unsigned __int8 *)v32[0];
      *(_QWORD *)(v15 + 48) = v32[0];
      if ( v29 )
        sub_1623210((__int64)v32, v29, v15 + 48);
    }
  }
  return v15;
}
