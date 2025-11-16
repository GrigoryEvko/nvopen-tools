// Function: sub_38EEC20
// Address: 0x38eec20
//
__int64 __fastcall sub_38EEC20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  _DWORD *v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // r13d
  _DWORD *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-70h] BYREF
  __int64 v22; // [rsp+8h] [rbp-68h] BYREF
  __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v25[2]; // [rsp+20h] [rbp-50h] BYREF
  char v26; // [rsp+30h] [rbp-40h]
  char v27; // [rsp+31h] [rbp-3Fh]

  v1 = sub_3909460(a1);
  v2 = sub_39092A0(v1);
  if ( (unsigned __int8)sub_38E31C0(a1, &v21, (__int64)".cv_inline_site_id", 18) )
    return 1;
  v27 = 1;
  v25[0] = "expected 'within' identifier in '.cv_inline_site_id' directive";
  v3 = *(_DWORD **)(a1 + 152);
  v26 = 3;
  if ( *v3 != 2 )
    goto LABEL_3;
  v10 = sub_3909460(a1);
  if ( *(_DWORD *)v10 == 2 )
  {
    v14 = *(_QWORD *)(v10 + 8);
    v13 = *(_QWORD *)(v10 + 16);
  }
  else
  {
    v11 = *(_QWORD *)(v10 + 16);
    if ( !v11 )
    {
LABEL_3:
      v4 = 1;
      goto LABEL_4;
    }
    v12 = v11 - 1;
    if ( v11 == 1 )
      v12 = 1;
    if ( v12 > v11 )
      v12 = *(_QWORD *)(v10 + 16);
    v13 = v12 - 1;
    v14 = *(_QWORD *)(v10 + 8) + 1LL;
  }
  if ( v13 != 6 )
    goto LABEL_3;
  if ( *(_DWORD *)v14 != 1752459639 )
    goto LABEL_3;
  v4 = 0;
  if ( *(_WORD *)(v14 + 4) != 28265 )
    goto LABEL_3;
LABEL_4:
  if ( (unsigned __int8)sub_3909CB0(a1, v4, v25) )
    return 1;
  sub_38EB180(a1);
  if ( (unsigned __int8)sub_38E31C0(a1, &v22, (__int64)".cv_inline_site_id", 18) )
    return 1;
  v27 = 1;
  v25[0] = "expected 'inlined_at' identifier in '.cv_inline_site_id' directive";
  v7 = *(_DWORD **)(a1 + 152);
  v26 = 3;
  if ( *v7 == 2 )
  {
    v15 = sub_3909460(a1);
    if ( *(_DWORD *)v15 == 2 )
    {
      v19 = *(_QWORD *)(v15 + 8);
      v18 = *(_QWORD *)(v15 + 16);
    }
    else
    {
      v16 = *(_QWORD *)(v15 + 16);
      if ( !v16 )
        goto LABEL_9;
      v17 = v16 - 1;
      if ( v16 == 1 )
        v17 = 1;
      if ( v17 > v16 )
        v17 = *(_QWORD *)(v15 + 16);
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v15 + 8) + 1LL;
    }
    if ( v18 == 10 && *(_QWORD *)v19 == 0x5F64656E696C6E69LL )
    {
      v8 = 0;
      if ( *(_WORD *)(v19 + 8) == 29793 )
        goto LABEL_10;
    }
  }
LABEL_9:
  v8 = 1;
LABEL_10:
  if ( (unsigned __int8)sub_3909CB0(a1, v8, v25) )
    return 1;
  sub_38EB180(a1);
  if ( (unsigned __int8)sub_38E3370(a1, &v23, (__int64)".cv_inline_site_id", 18) )
    return 1;
  v27 = 1;
  v26 = 3;
  v25[0] = "expected line number after 'inlined_at'";
  if ( (unsigned __int8)sub_3909D40(a1, &v24, v25) )
    return 1;
  if ( **(_DWORD **)(a1 + 152) == 4 )
  {
    v20 = sub_3909460(a1);
    v9 = *(_DWORD *)(v20 + 32) <= 0x40u ? *(_QWORD *)(v20 + 24) : **(_QWORD **)(v20 + 24);
    sub_38EB180(a1);
  }
  else
  {
    LODWORD(v9) = 0;
  }
  v27 = 1;
  v26 = 3;
  v25[0] = "unexpected token in '.cv_inline_site_id' directive";
  v5 = sub_3909E20(a1, 9, v25);
  if ( (_BYTE)v5 )
    return 1;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 328) + 616LL))(
          *(_QWORD *)(a1 + 328),
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
    return (unsigned int)sub_3909790(a1, v2, v25, 0, 0);
  }
  return v5;
}
