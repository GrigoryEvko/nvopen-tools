// Function: sub_1289360
// Address: 0x1289360
//
__int64 __fastcall sub_1289360(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v8; // rsi
  bool v9; // cc
  __int64 *v10; // r15
  char v11; // al
  __int64 *v12; // rbx
  _QWORD *v13; // r12
  __int64 *v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 *v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v34[2]; // [rsp+20h] [rbp-70h] BYREF
  char v35; // [rsp+30h] [rbp-60h]
  char v36; // [rsp+31h] [rbp-5Fh]
  _BYTE v37[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v38; // [rsp+50h] [rbp-40h]

  v6 = a3;
  v8 = *a2;
  if ( v8 != *(_QWORD *)a3 )
  {
    v9 = *(_BYTE *)(a3 + 16) <= 0x10u;
    v10 = *(__int64 **)(a1 + 8);
    v36 = 1;
    v34[0] = "sh_prom";
    v35 = 3;
    if ( v9 )
    {
      v6 = sub_15A4750(a3, v8, 0);
    }
    else
    {
      v38 = 257;
      v25 = sub_15FE0A0(a3, v8, 0, v37, 0);
      v26 = v10[1];
      v6 = v25;
      if ( v26 )
      {
        v32 = (__int64 *)v10[2];
        sub_157E9D0(v26 + 40, v25);
        v27 = *v32;
        v28 = *(_QWORD *)(v6 + 24) & 7LL;
        *(_QWORD *)(v6 + 32) = v32;
        v27 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v6 + 24) = v27 | v28;
        *(_QWORD *)(v27 + 8) = v6 + 24;
        *v32 = *v32 & 7 | (v6 + 24);
      }
      sub_164B780(v6, v34);
      v29 = *v10;
      if ( *v10 )
      {
        v33 = *v10;
        sub_1623A60(&v33, v29, 2);
        v30 = v6 + 48;
        if ( *(_QWORD *)(v6 + 48) )
        {
          sub_161E7C0(v6 + 48);
          v30 = v6 + 48;
        }
        v31 = v33;
        *(_QWORD *)(v6 + 48) = v33;
        if ( v31 )
          sub_1623210(&v33, v31, v30);
      }
    }
  }
  v11 = sub_127B3E0(a4);
  v36 = 1;
  v12 = *(__int64 **)(a1 + 8);
  v35 = 3;
  v34[0] = "shr";
  if ( v11 )
  {
    if ( *((_BYTE *)a2 + 16) <= 0x10u && *(_BYTE *)(v6 + 16) <= 0x10u )
      return sub_15A2D80(a2, v6, 0);
    v15 = a2;
    v38 = 257;
    v16 = 24;
    v17 = v6;
  }
  else
  {
    if ( *((_BYTE *)a2 + 16) <= 0x10u && *(_BYTE *)(v6 + 16) <= 0x10u )
      return sub_15A2DA0(a2, v6, 0);
    v17 = v6;
    v38 = 257;
    v15 = a2;
    v16 = 25;
  }
  v18 = sub_15FB440(v16, v15, v17, v37, 0);
  v19 = v12[1];
  v13 = (_QWORD *)v18;
  if ( v19 )
  {
    v20 = (unsigned __int64 *)v12[2];
    sub_157E9D0(v19 + 40, v18);
    v21 = v13[3];
    v22 = *v20;
    v13[4] = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v13 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v13 + 3);
  }
  sub_164B780(v13, v34);
  v23 = *v12;
  if ( *v12 )
  {
    v33 = *v12;
    sub_1623A60(&v33, v23, 2);
    if ( v13[6] )
      sub_161E7C0(v13 + 6);
    v24 = v33;
    v13[6] = v33;
    if ( v24 )
      sub_1623210(&v33, v24, v13 + 6);
  }
  return (__int64)v13;
}
