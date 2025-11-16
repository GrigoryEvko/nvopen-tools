// Function: sub_1B45090
// Address: 0x1b45090
//
__int64 __fastcall sub_1B45090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, unsigned int a6)
{
  __int64 v6; // r13
  __int64 v8; // rbx
  int v9; // eax
  int v10; // r14d
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int8 *v14; // rsi
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // r13
  unsigned __int64 *v20; // rbx
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v26; // r13
  _QWORD *v27; // rdi
  unsigned __int8 *v33; // [rsp+28h] [rbp-A8h] BYREF
  unsigned __int8 *v34[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v35; // [rsp+40h] [rbp-90h]
  unsigned __int8 *v36; // [rsp+50h] [rbp-80h] BYREF
  __int64 v37; // [rsp+58h] [rbp-78h]
  unsigned __int64 *v38; // [rsp+60h] [rbp-70h]
  __int64 v39; // [rsp+68h] [rbp-68h]
  __int64 v40; // [rsp+70h] [rbp-60h]
  int v41; // [rsp+78h] [rbp-58h]
  __int64 v42; // [rsp+80h] [rbp-50h]
  __int64 v43; // [rsp+88h] [rbp-48h]

  v6 = 0;
  v8 = a3;
  if ( a3 != a4 )
    v6 = a4;
  v9 = sub_15F4D60(a1);
  if ( v9 )
  {
    v10 = v9;
    v11 = 0;
    while ( 1 )
    {
      v12 = sub_15F4DF0(a1, v11);
      if ( v12 == v8 )
        break;
      if ( v12 == v6 )
      {
        ++v11;
        v6 = 0;
        if ( v11 == v10 )
          goto LABEL_10;
      }
      else
      {
        sub_157F2D0(v12, *(_QWORD *)(a1 + 40), 1);
LABEL_6:
        if ( ++v11 == v10 )
          goto LABEL_10;
      }
    }
    v8 = 0;
    goto LABEL_6;
  }
  v8 = a3;
LABEL_10:
  v13 = sub_16498A0(a1);
  v14 = *(unsigned __int8 **)(a1 + 48);
  v36 = 0;
  v39 = v13;
  v15 = *(_QWORD *)(a1 + 40);
  v40 = 0;
  v37 = v15;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v38 = (unsigned __int64 *)(a1 + 24);
  v34[0] = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)v34, (__int64)v14, 2);
    if ( v36 )
      sub_161E7C0((__int64)&v36, (__int64)v36);
    v36 = v34[0];
    if ( v34[0] )
      sub_1623210((__int64)v34, v34[0], (__int64)&v36);
  }
  v16 = *(unsigned __int8 **)(a1 + 48);
  v34[0] = v16;
  if ( v16 )
  {
    sub_1623A60((__int64)v34, (__int64)v16, 2);
    v17 = v36;
    if ( !v36 )
      goto LABEL_18;
  }
  else
  {
    v17 = v36;
    if ( !v36 )
      goto LABEL_20;
  }
  sub_161E7C0((__int64)&v36, (__int64)v17);
LABEL_18:
  v36 = v34[0];
  if ( v34[0] )
    sub_1623210((__int64)v34, v34[0], (__int64)&v36);
LABEL_20:
  if ( !(v6 | v8) )
  {
    if ( a3 != a4 )
    {
      v35 = 257;
      v18 = sub_1648A60(56, 3u);
      v19 = v18;
      if ( v18 )
        sub_15F83E0((__int64)v18, a3, a4, a2, 0);
      if ( v37 )
      {
        v20 = v38;
        sub_157E9D0(v37 + 40, (__int64)v19);
        v21 = v19[3];
        v22 = *v20;
        v19[4] = v20;
        v22 &= 0xFFFFFFFFFFFFFFF8LL;
        v19[3] = v22 | v21 & 7;
        *(_QWORD *)(v22 + 8) = v19 + 3;
        *v20 = *v20 & 7 | (unsigned __int64)(v19 + 3);
      }
      sub_164B780((__int64)v19, (__int64 *)v34);
      if ( v36 )
      {
        v33 = v36;
        sub_1623A60((__int64)&v33, (__int64)v36, 2);
        v23 = v19[6];
        if ( v23 )
          sub_161E7C0((__int64)(v19 + 6), v23);
        v24 = v33;
        v19[6] = v33;
        if ( v24 )
          sub_1623210((__int64)&v33, v24, (__int64)(v19 + 6));
      }
      if ( a5 != a6 )
        sub_1B423A0((__int64)v19, a5, a6);
      goto LABEL_38;
    }
LABEL_41:
    sub_1B44660((__int64 *)&v36, a3);
    goto LABEL_38;
  }
  if ( !v8 )
    goto LABEL_41;
  if ( v6 || a3 == a4 )
  {
    v26 = sub_16498A0(a1);
    v27 = sub_1648A60(56, 0);
    if ( v27 )
      sub_15F82A0((__int64)v27, v26, a1);
  }
  else
  {
    sub_1B44660((__int64 *)&v36, a4);
  }
LABEL_38:
  sub_1B44FE0(a1);
  if ( v36 )
    sub_161E7C0((__int64)&v36, (__int64)v36);
  return 1;
}
