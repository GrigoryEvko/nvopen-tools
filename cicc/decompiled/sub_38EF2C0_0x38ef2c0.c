// Function: sub_38EF2C0
// Address: 0x38ef2c0
//
char __fastcall sub_38EF2C0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r14
  int v6; // edx
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 (*v9)(); // rax
  char v10; // al
  char result; // al
  _DWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 (*v20)(); // r10
  __int64 v21; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v22; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v23; // [rsp+18h] [rbp-D8h]
  __int64 v24; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v25; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v26; // [rsp+38h] [rbp-B8h] BYREF
  const char *v27; // [rsp+40h] [rbp-B0h] BYREF
  char v28; // [rsp+50h] [rbp-A0h]
  char v29; // [rsp+51h] [rbp-9Fh]
  const char *v30; // [rsp+60h] [rbp-90h] BYREF
  char v31; // [rsp+70h] [rbp-80h]
  char v32; // [rsp+71h] [rbp-7Fh]
  _QWORD v33[2]; // [rsp+80h] [rbp-70h] BYREF
  char v34; // [rsp+90h] [rbp-60h]
  char v35; // [rsp+91h] [rbp-5Fh]
  _QWORD v36[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v37; // [rsp+B0h] [rbp-40h]
  int v38; // [rsp+B8h] [rbp-38h]

  v3 = a1[19];
  v25 = 0;
  v4 = sub_39092A0(v3);
  v36[0] = 0;
  v5 = v4;
  if ( sub_38EB6A0((__int64)a1, &v24, (__int64)v36) )
    return 1;
  v29 = 1;
  v6 = 0;
  v7 = a1[41];
  v27 = "expression is not a constant value";
  v8 = v24;
  v28 = 3;
  v9 = *(__int64 (**)())(*(_QWORD *)v7 + 72LL);
  if ( v9 != sub_168DB40 )
    v6 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v9)(v7, &v24, 0);
  v10 = sub_38CF2B0(v8, &v26, v6);
  if ( (unsigned __int8)sub_3909C80(a1, (unsigned __int8)v10 ^ 1u, v5, &v27) )
    return 1;
  v32 = 1;
  v30 = "expression is negative";
  v31 = 3;
  if ( (unsigned __int8)sub_3909C80(a1, v26 >> 63, v5, &v30) )
    return 1;
  v35 = 1;
  v34 = 3;
  v33[0] = "expected comma";
  if ( (unsigned __int8)sub_3909E20(a1, 25, v33) )
    return 1;
  v36[0] = "expected relocation name";
  LOWORD(v37) = 259;
  v12 = (_DWORD *)sub_3909460(a1);
  if ( (unsigned __int8)sub_3909CB0(a1, *v12 != 2, v36) )
    return 1;
  v13 = sub_39092A0(a1[19]);
  v14 = a1[19];
  v15 = v13;
  if ( *(_DWORD *)v14 == 2 )
  {
    v22 = *(_QWORD *)(v14 + 8);
    v23 = *(_QWORD *)(v14 + 16);
  }
  else
  {
    v23 = 0;
    v16 = *(_QWORD *)(v14 + 16);
    if ( v16 )
    {
      v17 = v16 - 1;
      if ( v16 == 1 )
        v17 = 1;
      if ( v17 <= v16 )
        v16 = v17;
      v23 = v16 - 1;
      v16 = 1;
    }
    v22 = *(_QWORD *)(v14 + 8) + v16;
  }
  sub_38EB180((__int64)a1);
  if ( *(_DWORD *)a1[19] == 25 )
  {
    sub_38EB180((__int64)a1);
    v21 = sub_3909290(a1 + 18);
    v36[0] = 0;
    result = sub_38EB6A0((__int64)a1, &v25, (__int64)v36);
    if ( result )
      return result;
    v36[0] = 0;
    v36[1] = 0;
    v37 = 0;
    v38 = 0;
    if ( !(unsigned __int8)sub_38CF2C0(v25, (__int64)v36, 0, 0) )
    {
      v35 = 1;
      v33[0] = "expression must be relocatable";
      v34 = 3;
      return sub_3909790(a1, v21, v33, 0, 0);
    }
  }
  LOWORD(v37) = 259;
  v36[0] = "unexpected token in .reloc directive";
  if ( (unsigned __int8)sub_3909E20(a1, 9, v36) )
    return 1;
  v18 = sub_390A040(a1[1]);
  v19 = a1[41];
  v20 = *(__int64 (**)())(*(_QWORD *)v19 + 992LL);
  if ( v20 == sub_168DC70
    || (result = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, unsigned __int64, __int64, __int64, __int64))v20)(
                   v19,
                   v24,
                   v22,
                   v23,
                   v25,
                   a2,
                   v18)) != 0 )
  {
    v36[0] = "unknown relocation name";
    LOWORD(v37) = 259;
    return sub_3909790(a1, v15, v36, 0, 0);
  }
  return result;
}
