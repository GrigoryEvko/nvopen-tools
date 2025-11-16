// Function: sub_1793E70
// Address: 0x1793e70
//
__int64 __fastcall sub_1793E70(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  bool v23; // zf
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  unsigned __int8 *v27; // [rsp+8h] [rbp-48h] BYREF
  char v28[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v29; // [rsp+20h] [rbp-30h]

  v11 = sub_15A1070(*(_QWORD *)a2, a3);
  v14 = v11;
  if ( *(_BYTE *)(v11 + 16) <= 0x10u )
  {
    if ( sub_1593BB0(v11, a3, v12, v13) )
      return a2;
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      a2 = sub_15A2D10((__int64 *)a2, v14, a5, a6, a7);
      v15 = sub_14DBA30(a2, *(_QWORD *)(a1 + 96), 0);
      if ( v15 )
        return v15;
      return a2;
    }
  }
  v29 = 257;
  v17 = sub_15FB440(27, (__int64 *)a2, v14, (__int64)v28, 0);
  v18 = *(_QWORD *)(a1 + 8);
  a2 = v17;
  if ( v18 )
  {
    v19 = *(__int64 **)(a1 + 16);
    sub_157E9D0(v18 + 40, v17);
    v20 = *(_QWORD *)(a2 + 24);
    v21 = *v19;
    *(_QWORD *)(a2 + 32) = v19;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 24) = v21 | v20 & 7;
    *(_QWORD *)(v21 + 8) = a2 + 24;
    *v19 = *v19 & 7 | (a2 + 24);
  }
  sub_164B780(a2, a4);
  v23 = *(_QWORD *)(a1 + 80) == 0;
  v27 = (unsigned __int8 *)a2;
  if ( v23 )
    sub_4263D6(a2, a4, v22);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v27);
  v24 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return a2;
  v27 = *(unsigned __int8 **)a1;
  sub_1623A60((__int64)&v27, v24, 2);
  v25 = *(_QWORD *)(a2 + 48);
  if ( v25 )
    sub_161E7C0(a2 + 48, v25);
  v26 = v27;
  *(_QWORD *)(a2 + 48) = v27;
  if ( !v26 )
    return a2;
  sub_1623210((__int64)&v27, v26, a2 + 48);
  return a2;
}
