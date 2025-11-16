// Function: sub_18252F0
// Address: 0x18252f0
//
__int64 __fastcall sub_18252F0(__int64 a1, __int64 a2, __int64 *a3, double a4, double a5, double a6)
{
  __int64 v9; // rsi
  __int64 v10; // r14
  unsigned __int8 v11; // al
  int v12; // r14d
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 *v20; // r14
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 *v37; // rbx
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rsi
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  unsigned __int8 *v43; // [rsp+8h] [rbp-78h] BYREF
  __int64 v44[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v45; // [rsp+20h] [rbp-60h]
  _BYTE v46[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v47; // [rsp+40h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 440);
  v45 = 257;
  v10 = sub_15A0680(*(_QWORD *)(a1 + 168), v9, 0);
  v11 = *(_BYTE *)(v10 + 16);
  if ( v11 <= 0x10u )
  {
    if ( v11 == 13 )
    {
      v17 = *(_DWORD *)(v10 + 32);
      if ( v17 > 0x40 )
      {
        if ( v17 == (unsigned int)sub_16A58F0(v10 + 24) )
          goto LABEL_5;
        if ( *(_BYTE *)(a2 + 16) <= 0x10u )
          goto LABEL_4;
        goto LABEL_17;
      }
      if ( *(_QWORD *)(v10 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17) )
        goto LABEL_5;
    }
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
LABEL_4:
      a2 = sub_15A2CF0((__int64 *)a2, v10, a4, a5, a6);
      goto LABEL_5;
    }
  }
LABEL_17:
  v47 = 257;
  v18 = sub_15FB440(26, (__int64 *)a2, v10, (__int64)v46, 0);
  v19 = a3[1];
  a2 = v18;
  if ( v19 )
  {
    v20 = (__int64 *)a3[2];
    sub_157E9D0(v19 + 40, v18);
    v21 = *(_QWORD *)(a2 + 24);
    v22 = *v20;
    *(_QWORD *)(a2 + 32) = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a2 + 24) = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = a2 + 24;
    *v20 = *v20 & 7 | (a2 + 24);
  }
  sub_164B780(a2, v44);
  v23 = *a3;
  if ( *a3 )
  {
    v43 = (unsigned __int8 *)*a3;
    sub_1623A60((__int64)&v43, v23, 2);
    v24 = *(_QWORD *)(a2 + 48);
    if ( v24 )
      sub_161E7C0(a2 + 48, v24);
    v25 = v43;
    *(_QWORD *)(a2 + 48) = v43;
    if ( v25 )
      sub_1623210((__int64)&v43, v25, a2 + 48);
  }
LABEL_5:
  v12 = dword_42B8E28[*(int *)(a1 + 156)];
  if ( v12 > 2 )
    v13 = *(_QWORD *)(a1 + 448) << v12;
  else
    v13 = *(_QWORD *)(a1 + 8LL * v12 + 448);
  v45 = 257;
  v14 = sub_15A0680(*(_QWORD *)(a1 + 168), v13, 0);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v14 + 16) <= 0x10u )
  {
    v15 = sub_15A2B30((__int64 *)a2, v14, 0, 0, a4, a5, a6);
    goto LABEL_10;
  }
  v47 = 257;
  v26 = sub_15FB440(11, (__int64 *)a2, v14, (__int64)v46, 0);
  v27 = a3[1];
  v15 = v26;
  if ( v27 )
  {
    v28 = (__int64 *)a3[2];
    sub_157E9D0(v27 + 40, v26);
    v29 = *(_QWORD *)(v15 + 24);
    v30 = *v28;
    *(_QWORD *)(v15 + 32) = v28;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v15 + 24) = v30 | v29 & 7;
    *(_QWORD *)(v30 + 8) = v15 + 24;
    *v28 = *v28 & 7 | (v15 + 24);
  }
  sub_164B780(v15, v44);
  v31 = *a3;
  if ( !*a3 )
    goto LABEL_10;
  v43 = (unsigned __int8 *)*a3;
  sub_1623A60((__int64)&v43, v31, 2);
  v32 = *(_QWORD *)(v15 + 48);
  if ( v32 )
    sub_161E7C0(v15 + 48, v32);
  v33 = v43;
  *(_QWORD *)(v15 + 48) = v43;
  if ( !v33 )
  {
LABEL_10:
    if ( v12 <= 0 )
      return v15;
    goto LABEL_31;
  }
  sub_1623210((__int64)&v43, v33, v15 + 48);
  if ( v12 <= 0 )
    return v15;
LABEL_31:
  v45 = 257;
  v34 = sub_15A0680(*(_QWORD *)v15, v12, 0);
  if ( *(_BYTE *)(v15 + 16) <= 0x10u && *(_BYTE *)(v34 + 16) <= 0x10u )
    return sub_15A2D80((__int64 *)v15, v34, 0, a4, a5, a6);
  v47 = 257;
  v35 = sub_15FB440(24, (__int64 *)v15, v34, (__int64)v46, 0);
  v36 = a3[1];
  v15 = v35;
  if ( v36 )
  {
    v37 = (__int64 *)a3[2];
    sub_157E9D0(v36 + 40, v35);
    v38 = *(_QWORD *)(v15 + 24);
    v39 = *v37;
    *(_QWORD *)(v15 + 32) = v37;
    v39 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v15 + 24) = v39 | v38 & 7;
    *(_QWORD *)(v39 + 8) = v15 + 24;
    *v37 = *v37 & 7 | (v15 + 24);
  }
  sub_164B780(v15, v44);
  v40 = *a3;
  if ( *a3 )
  {
    v43 = (unsigned __int8 *)*a3;
    sub_1623A60((__int64)&v43, v40, 2);
    v41 = *(_QWORD *)(v15 + 48);
    if ( v41 )
      sub_161E7C0(v15 + 48, v41);
    v42 = v43;
    *(_QWORD *)(v15 + 48) = v43;
    if ( v42 )
      sub_1623210((__int64)&v43, v42, v15 + 48);
  }
  return v15;
}
