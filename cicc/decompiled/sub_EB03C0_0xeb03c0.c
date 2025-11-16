// Function: sub_EB03C0
// Address: 0xeb03c0
//
__int64 __fastcall sub_EB03C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  __int64 v3; // rsi
  _DWORD *v4; // rax
  unsigned int v5; // r13d
  __int64 v7; // rsi
  _DWORD *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-80h] BYREF
  __int64 v22; // [rsp+8h] [rbp-78h] BYREF
  __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v25[4]; // [rsp+20h] [rbp-60h] BYREF
  char v26; // [rsp+40h] [rbp-40h]
  char v27; // [rsp+41h] [rbp-3Fh]

  v1 = sub_ECD7B0(a1);
  v2 = sub_ECD6A0(v1);
  if ( (unsigned __int8)sub_EA2660(a1, &v21) )
    return 1;
  v27 = 1;
  v3 = 1;
  v25[0] = "expected 'within' identifier in '.cv_inline_site_id' directive";
  v4 = *(_DWORD **)(a1 + 48);
  v26 = 3;
  if ( *v4 != 2 )
    goto LABEL_3;
  v10 = sub_ECD7B0(a1);
  if ( *(_DWORD *)v10 == 2 )
  {
    v13 = *(_QWORD *)(v10 + 8);
    v14 = *(_QWORD *)(v10 + 16);
  }
  else
  {
    v11 = *(_QWORD *)(v10 + 16);
    if ( !v11 )
    {
LABEL_21:
      v3 = 1;
      goto LABEL_3;
    }
    v12 = v11 - 1;
    if ( !v12 )
      v12 = 1;
    v13 = *(_QWORD *)(v10 + 8) + 1LL;
    v14 = v12 - 1;
  }
  if ( v14 != 6 )
    goto LABEL_21;
  if ( *(_DWORD *)v13 != 1752459639 )
    goto LABEL_21;
  v3 = 0;
  if ( *(_WORD *)(v13 + 4) != 28265 )
    goto LABEL_21;
LABEL_3:
  if ( (unsigned __int8)sub_ECE0A0(a1, v3, v25) )
    return 1;
  sub_EABFE0(a1);
  if ( (unsigned __int8)sub_EA2660(a1, &v22) )
    return 1;
  v27 = 1;
  v7 = 1;
  v25[0] = "expected 'inlined_at' identifier in '.cv_inline_site_id' directive";
  v8 = *(_DWORD **)(a1 + 48);
  v26 = 3;
  if ( *v8 != 2 )
    goto LABEL_8;
  v15 = sub_ECD7B0(a1);
  if ( *(_DWORD *)v15 == 2 )
  {
    v18 = *(_QWORD *)(v15 + 8);
    v19 = *(_QWORD *)(v15 + 16);
    goto LABEL_27;
  }
  v16 = *(_QWORD *)(v15 + 16);
  if ( v16 )
  {
    v17 = v16 - 1;
    if ( !v17 )
      v17 = 1;
    v18 = *(_QWORD *)(v15 + 8) + 1LL;
    v19 = v17 - 1;
LABEL_27:
    if ( v19 == 10 && *(_QWORD *)v18 == 0x5F64656E696C6E69LL )
    {
      v7 = 0;
      if ( *(_WORD *)(v18 + 8) == 29793 )
        goto LABEL_8;
    }
  }
  v7 = 1;
LABEL_8:
  if ( (unsigned __int8)sub_ECE0A0(a1, v7, v25) )
    return 1;
  sub_EABFE0(a1);
  if ( (unsigned __int8)sub_EA3FF0(a1, &v23, (__int64)".cv_inline_site_id", 18) )
    return 1;
  v27 = 1;
  v26 = 3;
  v25[0] = "expected line number after 'inlined_at'";
  if ( (unsigned __int8)sub_ECE130(a1, &v24, v25) )
    return 1;
  LODWORD(v9) = 0;
  if ( **(_DWORD **)(a1 + 48) == 4 )
  {
    v20 = sub_ECD7B0(a1);
    if ( *(_DWORD *)(v20 + 32) <= 0x40u )
      v9 = *(_QWORD *)(v20 + 24);
    else
      v9 = **(_QWORD **)(v20 + 24);
    sub_EABFE0(a1);
  }
  v5 = sub_ECE000(a1);
  if ( (_BYTE)v5 )
    return 1;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 728LL))(
          *(_QWORD *)(a1 + 232),
          (unsigned int)v21,
          (unsigned int)v22,
          (unsigned int)v23,
          v24,
          (unsigned int)v9,
          v2) )
  {
    v27 = 1;
    v25[0] = "function id already allocated";
    v26 = 3;
    return (unsigned int)sub_ECDA70(a1, v2, v25, 0, 0);
  }
  return v5;
}
