// Function: sub_1804A80
// Address: 0x1804a80
//
__int64 __fastcall sub_1804A80(__int64 a1, __int64 a2, __int64 *a3, double a4, double a5, double a6)
{
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // r14
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rcx
  unsigned __int8 *v30; // [rsp+8h] [rbp-78h] BYREF
  __int64 v31[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v32; // [rsp+20h] [rbp-60h]
  _BYTE v33[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v34; // [rsp+40h] [rbp-40h]

  v9 = *(int *)(a1 + 240);
  v10 = *(_QWORD *)a2;
  v32 = 257;
  v11 = sub_15A0680(v10, v9, 0);
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v11 + 16) > 0x10u )
  {
    v34 = 257;
    v17 = sub_15FB440(24, (__int64 *)a2, v11, (__int64)v33, 0);
    v18 = a3[1];
    v13 = v17;
    if ( v18 )
    {
      v19 = (__int64 *)a3[2];
      sub_157E9D0(v18 + 40, v17);
      v20 = *(_QWORD *)(v13 + 24);
      v21 = *v19;
      *(_QWORD *)(v13 + 32) = v19;
      v21 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v13 + 24) = v21 | v20 & 7;
      *(_QWORD *)(v21 + 8) = v13 + 24;
      *v19 = *v19 & 7 | (v13 + 24);
    }
    sub_164B780(v13, v31);
    v22 = *a3;
    if ( *a3 )
    {
      v30 = (unsigned __int8 *)*a3;
      sub_1623A60((__int64)&v30, v22, 2);
      v23 = *(_QWORD *)(v13 + 48);
      if ( v23 )
        sub_161E7C0(v13 + 48, v23);
      v24 = v30;
      *(_QWORD *)(v13 + 48) = v30;
      if ( v24 )
        sub_1623210((__int64)&v30, v24, v13 + 48);
    }
  }
  else
  {
    v13 = sub_15A2D80((__int64 *)a2, v11, 0, a4, a5, a6);
  }
  v14 = *(_QWORD *)(a1 + 248);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a1 + 720);
    if ( !v15 )
      v15 = sub_15A0680(*(_QWORD *)(a1 + 232), v14, 0);
    if ( !*(_BYTE *)(a1 + 256) )
    {
      v34 = 257;
      return sub_12899C0(a3, v13, v15, (__int64)v33, 0, 0);
    }
    v32 = 257;
    if ( *(_BYTE *)(v15 + 16) <= 0x10u )
    {
      if ( sub_1593BB0(v15, v14, v12, 257) )
        return v13;
      if ( *(_BYTE *)(v13 + 16) <= 0x10u )
        return sub_15A2D10((__int64 *)v13, v15, a4, a5, a6);
    }
    v34 = 257;
    v25 = sub_15FB440(27, (__int64 *)v13, v15, (__int64)v33, 0);
    v26 = a3[1];
    v13 = v25;
    if ( v26 )
    {
      v27 = (__int64 *)a3[2];
      sub_157E9D0(v26 + 40, v25);
      v28 = *(_QWORD *)(v13 + 24);
      v29 = *v27;
      *(_QWORD *)(v13 + 32) = v27;
      v29 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v13 + 24) = v29 | v28 & 7;
      *(_QWORD *)(v29 + 8) = v13 + 24;
      *v27 = *v27 & 7 | (v13 + 24);
    }
    sub_164B780(v13, v31);
    sub_12A86E0(a3, v13);
  }
  return v13;
}
