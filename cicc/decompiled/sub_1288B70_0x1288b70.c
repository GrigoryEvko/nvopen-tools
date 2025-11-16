// Function: sub_1288B70
// Address: 0x1288b70
//
__int64 __fastcall sub_1288B70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // rsi
  bool v7; // cc
  __int64 *v8; // r14
  __int64 *v9; // rbx
  _QWORD *v10; // r12
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h] BYREF
  _QWORD v27[2]; // [rsp+10h] [rbp-70h] BYREF
  char v28; // [rsp+20h] [rbp-60h]
  char v29; // [rsp+21h] [rbp-5Fh]
  _BYTE v30[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v31; // [rsp+40h] [rbp-40h]

  v4 = a3;
  v6 = *a2;
  if ( v6 != *(_QWORD *)a3 )
  {
    v7 = *(_BYTE *)(a3 + 16) <= 0x10u;
    v8 = *(__int64 **)(a1 + 8);
    v29 = 1;
    v27[0] = "sh_prom";
    v28 = 3;
    if ( v7 )
    {
      v4 = sub_15A4750(a3, v6, 0);
    }
    else
    {
      v31 = 257;
      v19 = sub_15FE0A0(a3, v6, 0, v30, 0);
      v20 = v8[1];
      v4 = v19;
      if ( v20 )
      {
        v21 = (__int64 *)v8[2];
        sub_157E9D0(v20 + 40, v19);
        v22 = *(_QWORD *)(v4 + 24);
        v23 = *v21;
        *(_QWORD *)(v4 + 32) = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v4 + 24) = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v4 + 24;
        *v21 = *v21 & 7 | (v4 + 24);
      }
      sub_164B780(v4, v27);
      v24 = *v8;
      if ( *v8 )
      {
        v26 = *v8;
        sub_1623A60(&v26, v24, 2);
        if ( *(_QWORD *)(v4 + 48) )
          sub_161E7C0(v4 + 48);
        v25 = v26;
        *(_QWORD *)(v4 + 48) = v26;
        if ( v25 )
          sub_1623210(&v26, v25, v4 + 48);
      }
    }
  }
  v7 = *((_BYTE *)a2 + 16) <= 0x10u;
  v9 = *(__int64 **)(a1 + 8);
  v29 = 1;
  v27[0] = "shl";
  v28 = 3;
  if ( v7 && *(_BYTE *)(v4 + 16) <= 0x10u )
    return sub_15A2D50(a2, v4, 0, 0);
  v31 = 257;
  v12 = sub_15FB440(23, a2, v4, v30, 0);
  v13 = v9[1];
  v10 = (_QWORD *)v12;
  if ( v13 )
  {
    v14 = (unsigned __int64 *)v9[2];
    sub_157E9D0(v13 + 40, v12);
    v15 = v10[3];
    v16 = *v14;
    v10[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v10 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780(v10, v27);
  v17 = *v9;
  if ( *v9 )
  {
    v26 = *v9;
    sub_1623A60(&v26, v17, 2);
    if ( v10[6] )
      sub_161E7C0(v10 + 6);
    v18 = v26;
    v10[6] = v26;
    if ( v18 )
      sub_1623210(&v26, v18, v10 + 6);
  }
  return (__int64)v10;
}
