// Function: sub_12812E0
// Address: 0x12812e0
//
_QWORD *__fastcall sub_12812E0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // r15
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned __int64 *v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 *v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // [rsp+0h] [rbp-90h]
  __int64 v29; // [rsp+18h] [rbp-78h] BYREF
  _BYTE v30[16]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v31; // [rsp+30h] [rbp-60h]
  _BYTE v32[16]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v33; // [rsp+50h] [rbp-40h]

  v3 = (_QWORD *)a2;
  v5 = a1[4];
  v31 = 257;
  v6 = *(_QWORD *)(v5 + 728);
  if ( v6 != *(_QWORD *)a2 )
  {
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v33 = 257;
      v20 = sub_15FDBD0(47, a2, v6, v32, 0);
      v21 = a1[7];
      v3 = (_QWORD *)v20;
      if ( v21 )
      {
        v22 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v21 + 40, v20);
        v23 = v3[3];
        v24 = *v22;
        v3[4] = v22;
        v24 &= 0xFFFFFFFFFFFFFFF8LL;
        v3[3] = v24 | v23 & 7;
        *(_QWORD *)(v24 + 8) = v3 + 3;
        *v22 = *v22 & 7 | (unsigned __int64)(v3 + 3);
      }
      sub_164B780(v3, v30);
      v25 = a1[6];
      if ( v25 )
      {
        v29 = a1[6];
        sub_1623A60(&v29, v25, 2);
        if ( v3[6] )
          sub_161E7C0(v3 + 6);
        v26 = v29;
        v3[6] = v29;
        if ( v26 )
          sub_1623210(&v29, v26, v3 + 6);
      }
    }
    else
    {
      v3 = (_QWORD *)sub_15A46C0(47, a2, v6, 0);
    }
  }
  v31 = 257;
  v33 = 257;
  v7 = sub_1648A60(56, 1);
  v8 = (_QWORD *)v7;
  if ( v7 )
  {
    v9 = v7;
    v27 = v7 - 24;
    sub_15F1EA0(v7, a3, 58, v7 - 24, 1, 0);
    if ( *(v8 - 3) )
    {
      v10 = *(v8 - 2);
      v11 = *(v8 - 1) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *(v8 - 3) = v3;
    if ( v3 )
    {
      v12 = v3[1];
      *(v8 - 2) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = (unsigned __int64)(v8 - 2) | *(_QWORD *)(v12 + 16) & 3LL;
      *(v8 - 1) = (unsigned __int64)(v3 + 1) | *(v8 - 1) & 3LL;
      v3[1] = v27;
    }
    sub_164B780(v8, v32);
  }
  else
  {
    v9 = 0;
  }
  v13 = a1[7];
  if ( v13 )
  {
    v14 = (unsigned __int64 *)a1[8];
    sub_157E9D0(v13 + 40, v8);
    v15 = v8[3];
    v16 = *v14;
    v8[4] = v14;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v8[3] = v16 | v15 & 7;
    *(_QWORD *)(v16 + 8) = v8 + 3;
    *v14 = *v14 & 7 | (unsigned __int64)(v8 + 3);
  }
  sub_164B780(v9, v30);
  v17 = a1[6];
  if ( v17 )
  {
    v29 = a1[6];
    sub_1623A60(&v29, v17, 2);
    if ( v8[6] )
      sub_161E7C0(v8 + 6);
    v18 = v29;
    v8[6] = v29;
    if ( v18 )
      sub_1623210(&v29, v18, v8 + 6);
  }
  return v8;
}
