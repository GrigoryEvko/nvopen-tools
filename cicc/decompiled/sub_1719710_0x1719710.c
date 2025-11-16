// Function: sub_1719710
// Address: 0x1719710
//
__int64 __fastcall sub_1719710(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v6; // r9
  bool v8; // cc
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // r13d
  __int64 v16; // rdi
  __int64 *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v25[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v26; // [rsp+20h] [rbp-60h]
  _QWORD v27[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v28; // [rsp+40h] [rbp-40h]

  v6 = a3;
  v8 = *(_BYTE *)(a2 + 16) <= 0x10u;
  v9 = *a1;
  v26 = 257;
  if ( !v8
    || *(_BYTE *)(a3 + 16) > 0x10u
    || (v10 = sub_15A2A30((__int64 *)0xC, (__int64 *)a2, a3, 0, 0, a4, a5, a6),
        (v11 = sub_14DBA30(v10, *(_QWORD *)(v9 + 96), 0)) == 0)
    && (v6 = a3, (v11 = v10) == 0) )
  {
    v28 = 257;
    v13 = sub_15FB440(12, (__int64 *)a2, v6, (__int64)v27, 0);
    v14 = *(_QWORD *)(v9 + 32);
    v15 = *(_DWORD *)(v9 + 40);
    v11 = v13;
    if ( v14 )
      sub_1625C10(v13, 3, v14);
    sub_15F2440(v11, v15);
    v16 = *(_QWORD *)(v9 + 8);
    if ( v16 )
    {
      v17 = *(__int64 **)(v9 + 16);
      sub_157E9D0(v16 + 40, v11);
      v18 = *(_QWORD *)(v11 + 24);
      v19 = *v17;
      *(_QWORD *)(v11 + 32) = v17;
      v19 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v11 + 24) = v19 | v18 & 7;
      *(_QWORD *)(v19 + 8) = v11 + 24;
      *v17 = *v17 & 7 | (v11 + 24);
    }
    sub_164B780(v11, v25);
    v27[0] = v11;
    if ( !*(_QWORD *)(v9 + 80) )
      sub_4263D6(v11, v25, v20);
    (*(void (__fastcall **)(__int64, _QWORD *))(v9 + 88))(v9 + 64, v27);
    v21 = *(_QWORD *)v9;
    if ( *(_QWORD *)v9 )
    {
      v27[0] = *(_QWORD *)v9;
      sub_1623A60((__int64)v27, v21, 2);
      v22 = *(_QWORD *)(v11 + 48);
      if ( v22 )
        sub_161E7C0(v11 + 48, v22);
      v23 = (unsigned __int8 *)v27[0];
      *(_QWORD *)(v11 + 48) = v27[0];
      if ( v23 )
        sub_1623210((__int64)v27, v23, v11 + 48);
    }
  }
  if ( *(_BYTE *)(v11 + 16) > 0x17u )
    sub_1718FD0((__int64)a1, v11);
  return v11;
}
