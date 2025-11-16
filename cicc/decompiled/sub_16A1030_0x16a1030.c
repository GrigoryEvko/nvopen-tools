// Function: sub_16A1030
// Address: 0x16a1030
//
__int64 __fastcall sub_16A1030(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  void *v5; // r15
  void *v6; // r14
  void *v7; // rax
  __int64 *v8; // r8
  __int64 v9; // r9
  void *v10; // r12
  _BYTE *v12; // rax
  char v13; // al
  char v14; // al
  char v15; // al
  __int64 v16; // rsi
  char v17; // al
  char v18; // al
  unsigned int v19; // r12d
  __int64 *v20; // [rsp+0h] [rbp-80h]
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 *v22; // [rsp+8h] [rbp-78h]
  _BYTE v23[8]; // [rsp+10h] [rbp-70h] BYREF
  void *v24[3]; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v25[8]; // [rsp+30h] [rbp-50h] BYREF
  void *v26[9]; // [rsp+38h] [rbp-48h] BYREF

  v5 = *(void **)(a1 + 8);
  v6 = sub_1698270();
  v7 = sub_16982C0();
  v8 = (__int64 *)(a1 + 8);
  v9 = a2 + 8;
  v10 = v7;
  if ( v6 != v5 )
    goto LABEL_2;
  v12 = (_BYTE *)sub_16D40F0(qword_4FBB490);
  v9 = a2 + 8;
  v8 = (__int64 *)(a1 + 8);
  v13 = v12 ? *v12 : LOBYTE(qword_4FBB490[2]);
  v5 = *(void **)(a1 + 8);
  if ( !v13 )
    goto LABEL_2;
  if ( v5 == v10 )
  {
    v14 = sub_16A0F40((__int64)v8, a2, a3, a4, a5);
    v9 = a2 + 8;
    v8 = (__int64 *)(a1 + 8);
  }
  else
  {
    v14 = sub_16984B0((__int64)v8);
    v8 = (__int64 *)(a1 + 8);
    v9 = a2 + 8;
  }
  if ( !v14 )
  {
    v20 = v8;
    v21 = v9;
    if ( *(void **)(a2 + 8) == v10 )
    {
      v15 = sub_16A0F40(v9, a2, a3, a4, a5);
      v8 = v20;
      v9 = v21;
    }
    else
    {
      v15 = sub_16984B0(v9);
      v9 = v21;
      v8 = v20;
    }
    if ( !v15 )
    {
      v5 = *(void **)(a1 + 8);
LABEL_2:
      if ( v5 == v10 )
        return sub_16A1240(v8, v9);
      else
        return sub_1698CF0((__int64)v8, v9);
    }
  }
  v22 = (__int64 *)v9;
  sub_169C7A0(v24, v8);
  v16 = (__int64)v22;
  sub_169C7A0(v26, v22);
  if ( v24[0] == v10 )
    v17 = sub_16A0F40((__int64)v24, (__int64)v22, a3, a4, a5);
  else
    v17 = sub_16984B0((__int64)v24);
  if ( v17 )
  {
    v16 = 0;
    if ( v24[0] == v10 )
      sub_169C980(v24, 0);
    else
      sub_169B620((__int64)v24, 0);
  }
  if ( v26[0] == v10 )
    v18 = sub_16A0F40((__int64)v26, v16, a3, a4, a5);
  else
    v18 = sub_16984B0((__int64)v26);
  if ( v18 )
  {
    if ( v10 == v26[0] )
      sub_169C980(v26, 0);
    else
      sub_169B620((__int64)v26, 0);
  }
  v19 = sub_16A1030(v23, v25);
  sub_127D120(v26);
  sub_127D120(v24);
  return v19;
}
