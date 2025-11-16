// Function: sub_2450220
// Address: 0x2450220
//
__int64 __fastcall sub_2450220(__int64 **a1, __int64 a2, int a3)
{
  __int64 *v4; // r13
  __int64 v5; // rax
  __int64 *v6; // r14
  int v7; // ecx
  unsigned __int64 v8; // rax
  const char *v9; // rsi
  unsigned __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v12[10]; // [rsp+10h] [rbp-50h] BYREF

  v4 = *a1;
  v5 = sub_BCB120(*a1);
  v11 = 0;
  v6 = (__int64 *)v5;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 168LL) )
  {
    v7 = 79;
  }
  else
  {
    v7 = 54;
    if ( !*(_BYTE *)(*(_QWORD *)a2 + 170LL) )
      goto LABEL_3;
  }
  v11 = sub_A7A090((__int64 *)&v11, *a1, 3, v7);
LABEL_3:
  v12[0] = sub_BCB2E0(v4);
  v12[1] = sub_BCE3C0(v4, 0);
  v12[2] = sub_BCB2D0(v4);
  v8 = sub_BCF480(v6, v12, 3, 0);
  v9 = "__llvm_profile_instrument_memop";
  if ( !a3 )
    v9 = "__llvm_profile_instrument_target";
  return sub_BA8C10((__int64)a1, (__int64)v9, (a3 == 0) + 31LL, v8, v11);
}
