// Function: sub_38F3A80
// Address: 0x38f3a80
//
__int64 __fastcall sub_38F3A80(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // rdx
  unsigned int v6; // ecx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  unsigned int v9; // ecx
  unsigned __int8 v10; // al
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v17; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v18[2]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v19[2]; // [rsp+20h] [rbp-C0h] BYREF
  const char *v20; // [rsp+30h] [rbp-B0h] BYREF
  char v21; // [rsp+40h] [rbp-A0h]
  char v22; // [rsp+41h] [rbp-9Fh]
  const char *v23; // [rsp+50h] [rbp-90h] BYREF
  char v24; // [rsp+60h] [rbp-80h]
  char v25; // [rsp+61h] [rbp-7Fh]
  const char *v26; // [rsp+70h] [rbp-70h] BYREF
  char v27; // [rsp+80h] [rbp-60h]
  char v28; // [rsp+81h] [rbp-5Fh]
  _QWORD v29[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v30; // [rsp+A0h] [rbp-40h]

  v18[0] = 0;
  v18[1] = 0;
  v19[0] = 0;
  v19[1] = 0;
  v2 = sub_3909460(a1);
  v17 = sub_39092A0(v2);
  if ( (unsigned __int8)sub_38E31C0(a1, &v16, (__int64)".cv_linetable", 13) )
    return 1;
  v22 = 1;
  v20 = "unexpected token in '.cv_linetable' directive";
  v21 = 3;
  if ( (unsigned __int8)sub_3909E20(a1, 25, &v20) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v17) )
    return 1;
  v25 = 1;
  v23 = "expected identifier in directive";
  v24 = 3;
  v7 = sub_38F0EE0(a1, v18, v5, v6);
  if ( (unsigned __int8)sub_3909C80(a1, v7, v17, &v23) )
    return 1;
  v28 = 1;
  v26 = "unexpected token in '.cv_linetable' directive";
  v27 = 3;
  if ( (unsigned __int8)sub_3909E20(a1, 25, &v26) )
    return 1;
  if ( (unsigned __int8)sub_3909470(a1, &v17) )
    return 1;
  v29[0] = "expected identifier in directive";
  v30 = 259;
  v10 = sub_38F0EE0(a1, v19, v8, v9);
  v3 = sub_3909C80(a1, v10, v17, v29);
  if ( (_BYTE)v3 )
  {
    return 1;
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 320);
    v29[0] = v18;
    v30 = 261;
    v12 = sub_38BF510(v11, (__int64)v29);
    v13 = *(_QWORD *)(a1 + 320);
    v29[0] = v19;
    v14 = v12;
    v30 = 261;
    v15 = sub_38BF510(v13, (__int64)v29);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 632LL))(
      *(_QWORD *)(a1 + 328),
      (unsigned int)v16,
      v14,
      v15);
  }
  return v3;
}
