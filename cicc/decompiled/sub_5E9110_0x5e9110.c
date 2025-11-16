// Function: sub_5E9110
// Address: 0x5e9110
//
__int64 __fastcall sub_5E9110(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 i; // r13
  __int64 j; // r12
  int v8; // eax
  _BOOL4 v9; // r9d
  int v10; // eax
  __int64 v11; // r8
  unsigned int v12; // r9d
  int v13; // eax
  __int64 v14; // rcx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r11
  __int64 v20; // rsi
  _BYTE *v21; // rcx
  __int64 *v22; // rdi
  __int64 v23; // rdx
  int v24; // eax
  _BOOL4 v25; // eax
  int v26; // eax
  __int64 v27; // r11
  _BOOL4 v28; // r9d
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rax
  _BOOL4 v32; // [rsp+Ch] [rbp-64h]
  _BOOL4 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  bool v36; // [rsp+20h] [rbp-50h]
  unsigned int v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  _BOOL4 v39; // [rsp+20h] [rbp-50h]
  __int64 v41; // [rsp+30h] [rbp-40h] BYREF
  __int64 v42[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 88);
  v5 = *(_QWORD *)(a2 + 88);
  for ( i = *(_QWORD *)(v4 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_QWORD *)(v5 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v36 = (*(_BYTE *)(a1 + 81) & 0x10) != 0;
  *a3 = 0;
  v8 = sub_8DE890(i, j, 0x200000, 0);
  v9 = v36;
  if ( !v8 )
    return 0;
  if ( !v36 )
    goto LABEL_7;
  v16 = sub_8D7820(i, j, 0, 0);
  v9 = v36;
  if ( v16 )
    goto LABEL_7;
  v33 = v36;
  v38 = sub_5E65C0(a1, &v41);
  v17 = sub_5E65C0(a2, v42);
  v19 = v38;
  v9 = v33;
  v20 = v17;
  if ( !v38 || !v17 )
  {
    v25 = v38 == v17;
LABEL_21:
    if ( !v25 )
      return 0;
    goto LABEL_7;
  }
  v21 = *(_BYTE **)(v41 + 168);
  v22 = *(__int64 **)(v42[0] + 168);
  v23 = *v22;
  if ( *(_QWORD *)v21 && (*(_BYTE *)(*(_QWORD *)v21 + 35LL) & 1) != 0 )
  {
    if ( v23 && (*(_BYTE *)(v23 + 35) & 1) != 0 || (*((_BYTE *)v22 + 19) & 0xC0) != 0 )
      goto LABEL_19;
  }
  else if ( !v23 || (*(_BYTE *)(v23 + 35) & 1) == 0 || (v21[19] & 0xC0) != 0 )
  {
    goto LABEL_19;
  }
  v26 = sub_8D32E0(v38);
  v27 = v38;
  v28 = v33;
  if ( v26 )
  {
    v31 = sub_8D46C0(v38);
    v28 = v33;
    v27 = v31;
  }
  v32 = v28;
  v34 = v27;
  v29 = sub_8D32E0(v20);
  v19 = v34;
  v9 = v32;
  if ( v29 )
  {
    v30 = sub_8D46C0(v20);
    v9 = v32;
    v19 = v34;
    v20 = v30;
  }
LABEL_19:
  if ( v19 != v20 )
  {
    v39 = v9;
    v24 = sub_8D97D0(v19, v20, 0, v21, v18);
    v9 = v39;
    v25 = v24 != 0;
    goto LABEL_21;
  }
LABEL_7:
  v37 = v9;
  v10 = sub_739400(*(_QWORD *)(v4 + 216), *(_QWORD *)(v5 + 216));
  v12 = v37;
  if ( !v10 )
    return 0;
  v13 = *(unsigned __int8 *)(*(_QWORD *)(i + 168) + 20LL);
  v14 = v13 ^ (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(j + 168) + 20LL);
  if ( (((unsigned __int8)v13 ^ *(_BYTE *)(*(_QWORD *)(j + 168) + 20LL)) & 2) != 0 )
    return 0;
  if ( (v13 & 2) != 0 )
  {
    if ( !(unsigned int)sub_8D73A0(i, j) )
      return 0;
    v12 = v37;
  }
  if ( v12 )
    return v12;
  if ( !dword_4F077BC )
  {
LABEL_28:
    v12 = 1;
    *a3 = 1;
    return v12;
  }
  if ( qword_4F077A8 > 0x76BFu )
  {
    if ( sub_5E9060(*(_QWORD *)(a2 + 88), a1, dword_4F077BC, v14, v11) )
      return 0;
    if ( !dword_4F077BC )
      goto LABEL_28;
  }
  if ( (*(_BYTE *)(a2 + 81) & 2) == 0 )
    goto LABEL_28;
  return 0;
}
