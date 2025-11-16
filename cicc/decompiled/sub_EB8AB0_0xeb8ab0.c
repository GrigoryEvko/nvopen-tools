// Function: sub_EB8AB0
// Address: 0xeb8ab0
//
__int64 __fastcall sub_EB8AB0(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-B8h] BYREF
  const char *v14; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-A8h]
  const char *v16; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-98h]
  const char *v18; // [rsp+30h] [rbp-90h] BYREF
  char v19; // [rsp+50h] [rbp-70h]
  char v20; // [rsp+51h] [rbp-6Fh]
  const char *v21; // [rsp+60h] [rbp-60h] BYREF
  __int64 v22; // [rsp+68h] [rbp-58h]
  __int16 v23; // [rsp+80h] [rbp-40h]

  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v2 = sub_ECD7B0(a1);
  v13 = sub_ECD6A0(v2);
  if ( (unsigned __int8)sub_EA2660(a1, &v12) )
    return 1;
  v23 = 259;
  v21 = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v21) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v13) )
    return 1;
  v20 = 1;
  v18 = "expected identifier in directive";
  v19 = 3;
  v5 = sub_EB61F0(a1, (__int64 *)&v14);
  if ( (unsigned __int8)sub_ECE070(a1, v5, v13, &v18) )
    return 1;
  v21 = "expected comma";
  v23 = 259;
  if ( (unsigned __int8)sub_ECE210(a1, 26, &v21) )
    return 1;
  if ( (unsigned __int8)sub_ECD7C0(a1, &v13) )
    return 1;
  v21 = "expected identifier in directive";
  v23 = 259;
  v6 = sub_EB61F0(a1, (__int64 *)&v16);
  v3 = sub_ECE070(a1, v6, v13, &v21);
  if ( (_BYTE)v3 )
  {
    return 1;
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 224);
    v23 = 261;
    v21 = v14;
    v22 = v15;
    v8 = sub_E6C460(v7, &v21);
    v9 = *(_QWORD *)(a1 + 224);
    v10 = v8;
    v23 = 261;
    v21 = v16;
    v22 = v17;
    v11 = sub_E6C460(v9, &v21);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 232) + 744LL))(
      *(_QWORD *)(a1 + 232),
      (unsigned int)v12,
      v10,
      v11);
  }
  return v3;
}
