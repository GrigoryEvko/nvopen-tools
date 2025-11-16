// Function: sub_156DA60
// Address: 0x156da60
//
__int64 __fastcall sub_156DA60(__int64 *a1, unsigned int a2, _QWORD *a3, __int64 *a4)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  char v9; // al
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  char v15; // al
  bool v16; // cc
  __int64 v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rsi
  __int64 v27; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v28[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v29; // [rsp+30h] [rbp-60h]
  _QWORD v30[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v31; // [rsp+50h] [rbp-40h]

  v6 = sub_1643350(a1[3]);
  v7 = sub_16463B0(*a3, a2);
  v8 = sub_1599EF0(v7);
  v9 = *((_BYTE *)a4 + 16);
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v30[0] = ".splatinsert";
      v31 = 259;
    }
    else
    {
      if ( *((_BYTE *)a4 + 17) == 1 )
      {
        v10 = (__int64 *)*a4;
      }
      else
      {
        v10 = a4;
        v9 = 2;
      }
      v30[0] = v10;
      v30[1] = ".splatinsert";
      LOBYTE(v31) = v9;
      HIBYTE(v31) = 3;
    }
  }
  else
  {
    v31 = 256;
  }
  v11 = sub_15A0680(v6, 0, 0);
  v12 = sub_156D8B0(a1, v8, (__int64)a3, v11, (__int64)v30);
  v13 = sub_16463B0(v6, a2);
  v14 = sub_1598F00(v13);
  v15 = *((_BYTE *)a4 + 16);
  if ( !v15 )
  {
    v29 = 256;
    goto LABEL_23;
  }
  if ( v15 == 1 )
  {
    v28[0] = ".splat";
    v29 = 259;
LABEL_23:
    if ( *(_BYTE *)(v12 + 16) > 0x10u )
      goto LABEL_12;
    goto LABEL_24;
  }
  if ( *((_BYTE *)a4 + 17) == 1 )
    a4 = (__int64 *)*a4;
  else
    v15 = 2;
  LOBYTE(v29) = v15;
  HIBYTE(v29) = 3;
  v16 = *(_BYTE *)(v12 + 16) <= 0x10u;
  v28[0] = a4;
  v28[1] = ".splat";
  if ( !v16 )
    goto LABEL_12;
LABEL_24:
  if ( *(_BYTE *)(v8 + 16) <= 0x10u && *(_BYTE *)(v14 + 16) <= 0x10u )
    return sub_15A3950(v12, v8, v14, 0);
LABEL_12:
  v31 = 257;
  v17 = sub_1648A60(56, 3);
  v18 = (_QWORD *)v17;
  if ( v17 )
    sub_15FA660(v17, v12, v8, v14, v30, 0);
  v19 = a1[1];
  if ( v19 )
  {
    v20 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v19 + 40, v18);
    v21 = v18[3];
    v22 = *v20;
    v18[4] = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    v18[3] = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v18 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v18 + 3);
  }
  sub_164B780(v18, v28);
  v23 = *a1;
  if ( *a1 )
  {
    v27 = *a1;
    sub_1623A60(&v27, v23, 2);
    if ( v18[6] )
      sub_161E7C0(v18 + 6);
    v24 = v27;
    v18[6] = v27;
    if ( v24 )
      sub_1623210(&v27, v24, v18 + 6);
  }
  return (__int64)v18;
}
