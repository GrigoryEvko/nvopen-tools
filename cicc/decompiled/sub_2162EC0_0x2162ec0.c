// Function: sub_2162EC0
// Address: 0x2162ec0
//
__int64 __fastcall sub_2162EC0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  void *v4; // rbx
  int v5; // eax
  _WORD *v6; // rdx
  bool v7; // zf
  __int64 v8; // rax
  __int16 *v9; // rax
  int v10; // r13d
  __int16 *v11; // rax
  __int64 v12; // rax
  __int64 result; // rax
  __int16 *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rsi
  __int64 v17; // rbx
  bool v18; // [rsp+Fh] [rbp-71h] BYREF
  __int64 *v19; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-68h]
  __int64 v21[2]; // [rsp+20h] [rbp-60h] BYREF
  int v22; // [rsp+30h] [rbp-50h]
  __int16 v23; // [rsp+34h] [rbp-4Ch]
  char v24; // [rsp+36h] [rbp-4Ah]
  _BYTE v25[8]; // [rsp+40h] [rbp-40h] BYREF
  void *v26; // [rsp+48h] [rbp-38h] BYREF
  __int64 v27; // [rsp+50h] [rbp-30h]

  v3 = (__int64 *)(a1 + 40);
  v4 = sub_16982C0();
  if ( *(void **)(a1 + 40) == v4 )
    sub_169C6E0(&v26, (__int64)v3);
  else
    sub_16986C0(&v26, v3);
  v5 = *(_DWORD *)(a1 + 24);
  v6 = *(_WORD **)(a2 + 24);
  if ( v5 == 2 )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v6 <= 1u )
    {
      sub_16E7EE0(a2, "0f", 2u);
    }
    else
    {
      *v6 = 26160;
      *(_QWORD *)(a2 + 24) += 2LL;
    }
    v14 = (__int16 *)sub_1698270();
    v10 = 8;
    sub_16A3360((__int64)v25, v14, 0, &v18);
  }
  else
  {
    v7 = v5 == 3;
    v8 = *(_QWORD *)(a2 + 16);
    if ( v7 )
    {
      if ( (unsigned __int64)(v8 - (_QWORD)v6) <= 1 )
      {
        sub_16E7EE0(a2, "0d", 2u);
      }
      else
      {
        *v6 = 25648;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      v9 = (__int16 *)sub_1698280();
      v10 = 16;
      sub_16A3360((__int64)v25, v9, 0, &v18);
      if ( v4 == v26 )
        goto LABEL_8;
      goto LABEL_13;
    }
    if ( (unsigned __int64)(v8 - (_QWORD)v6) <= 1 )
    {
      sub_16E7EE0(a2, "0x", 2u);
    }
    else
    {
      *v6 = 30768;
      *(_QWORD *)(a2 + 24) += 2LL;
    }
    v11 = (__int16 *)sub_1698260();
    v10 = 4;
    sub_16A3360((__int64)v25, v11, 0, &v18);
  }
  if ( v4 == v26 )
  {
LABEL_8:
    sub_169D930((__int64)&v19, (__int64)&v26);
    goto LABEL_14;
  }
LABEL_13:
  sub_169D7E0((__int64)&v19, (__int64 *)&v26);
LABEL_14:
  v12 = (__int64)v19;
  if ( v20 > 0x40 )
    v12 = *v19;
  v21[0] = v12;
  v21[1] = 0;
  v22 = v10;
  v23 = 257;
  v24 = 0;
  result = sub_16E87C0(a2, v21);
  if ( v20 > 0x40 && v19 )
    result = j_j___libc_free_0_0(v19);
  if ( v4 != v26 )
    return sub_1698460((__int64)&v26);
  v15 = v27;
  if ( v27 )
  {
    v16 = 32LL * *(_QWORD *)(v27 - 8);
    v17 = v27 + v16;
    if ( v27 != v27 + v16 )
    {
      do
      {
        v17 -= 32;
        sub_127D120((_QWORD *)(v17 + 8));
      }
      while ( v15 != v17 );
    }
    return j_j_j___libc_free_0_0(v15 - 8);
  }
  return result;
}
