// Function: sub_182C3C0
// Address: 0x182c3c0
//
__int64 __fastcall sub_182C3C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // r15
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rsi
  unsigned __int8 *v31; // [rsp+18h] [rbp-78h] BYREF
  __int64 v32[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v33; // [rsp+30h] [rbp-60h]
  _BYTE v34[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v35; // [rsp+50h] [rbp-40h]

  v33 = 257;
  v10 = sub_15A0680(*(_QWORD *)a2, *(int *)(a1 + 224), 0);
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v10 + 16) > 0x10u )
  {
    v35 = 257;
    v22 = sub_15FB440(24, (__int64 *)a2, v10, (__int64)v34, 0);
    v23 = a4[1];
    v11 = v22;
    if ( v23 )
    {
      v24 = (__int64 *)a4[2];
      sub_157E9D0(v23 + 40, v22);
      v25 = *(_QWORD *)(v11 + 24);
      v26 = *v24;
      *(_QWORD *)(v11 + 32) = v24;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v11 + 24) = v26 | v25 & 7;
      *(_QWORD *)(v26 + 8) = v11 + 24;
      *v24 = *v24 & 7 | (v11 + 24);
    }
    sub_164B780(v11, v32);
    v27 = *a4;
    if ( *a4 )
    {
      v31 = (unsigned __int8 *)*a4;
      sub_1623A60((__int64)&v31, v27, 2);
      v28 = *(_QWORD *)(v11 + 48);
      v29 = v11 + 48;
      if ( v28 )
      {
        sub_161E7C0(v11 + 48, v28);
        v29 = v11 + 48;
      }
      v30 = v31;
      *(_QWORD *)(v11 + 48) = v31;
      if ( v30 )
        sub_1623210((__int64)&v31, v30, v29);
    }
  }
  else
  {
    v11 = sub_15A2D80((__int64 *)a2, v10, 0, a5, a6, a7);
  }
  if ( *(_QWORD *)(a1 + 232) )
  {
    v12 = *(_QWORD *)(a1 + 400);
    if ( !v12 )
      v12 = sub_15A0680(a3, *(_QWORD *)(a1 + 232), 0);
    v33 = 257;
    if ( *(_BYTE *)(v11 + 16) > 0x10u || *(_BYTE *)(v12 + 16) > 0x10u )
    {
      v35 = 257;
      v14 = sub_15FB440(11, (__int64 *)v11, v12, (__int64)v34, 0);
      v15 = a4[1];
      v11 = v14;
      if ( v15 )
      {
        v16 = (__int64 *)a4[2];
        sub_157E9D0(v15 + 40, v14);
        v17 = *(_QWORD *)(v11 + 24);
        v18 = *v16;
        *(_QWORD *)(v11 + 32) = v16;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v11 + 24) = v18 | v17 & 7;
        *(_QWORD *)(v18 + 8) = v11 + 24;
        *v16 = *v16 & 7 | (v11 + 24);
      }
      sub_164B780(v11, v32);
      v19 = *a4;
      if ( *a4 )
      {
        v31 = (unsigned __int8 *)*a4;
        sub_1623A60((__int64)&v31, v19, 2);
        v20 = *(_QWORD *)(v11 + 48);
        if ( v20 )
          sub_161E7C0(v11 + 48, v20);
        v21 = v31;
        *(_QWORD *)(v11 + 48) = v31;
        if ( v21 )
          sub_1623210((__int64)&v31, v21, v11 + 48);
      }
    }
    else
    {
      return sub_15A2B30((__int64 *)v11, v12, 0, 0, a5, a6, a7);
    }
  }
  return v11;
}
