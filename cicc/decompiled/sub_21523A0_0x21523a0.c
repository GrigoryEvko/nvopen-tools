// Function: sub_21523A0
// Address: 0x21523a0
//
__int64 __fastcall sub_21523A0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v5; // rsi
  void *v6; // rbx
  __int16 *v7; // rax
  char *v8; // r15
  int v9; // r13d
  _WORD *v10; // rdx
  __int64 v11; // rax
  __int64 result; // rax
  __int16 *v13; // rax
  __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rbx
  bool v17; // [rsp+Fh] [rbp-81h] BYREF
  __int64 *v18; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-78h]
  __int64 v20[2]; // [rsp+20h] [rbp-70h] BYREF
  int v21; // [rsp+30h] [rbp-60h]
  __int16 v22; // [rsp+34h] [rbp-5Ch]
  char v23; // [rsp+36h] [rbp-5Ah]
  _BYTE v24[8]; // [rsp+40h] [rbp-50h] BYREF
  void *v25; // [rsp+48h] [rbp-48h] BYREF
  __int64 v26; // [rsp+50h] [rbp-40h]

  v5 = a2 + 4;
  v6 = sub_16982C0();
  if ( (void *)a2[4] == v6 )
  {
    sub_169C6E0(&v25, (__int64)v5);
    if ( *(_BYTE *)(*a2 + 8) != 2 )
      goto LABEL_3;
  }
  else
  {
    sub_16986C0(&v25, v5);
    if ( *(_BYTE *)(*a2 + 8) != 2 )
    {
LABEL_3:
      v7 = (__int16 *)sub_1698280();
      v8 = "0d";
      v9 = 16;
      sub_16A3360((__int64)v24, v7, 0, &v17);
      goto LABEL_4;
    }
  }
  v13 = (__int16 *)sub_1698270();
  v8 = "0f";
  v9 = 8;
  sub_16A3360((__int64)v24, v13, 0, &v17);
LABEL_4:
  if ( v6 == v25 )
    sub_169D930((__int64)&v18, (__int64)&v25);
  else
    sub_169D7E0((__int64)&v18, (__int64 *)&v25);
  v10 = *(_WORD **)(a3 + 24);
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v10 > 1u )
  {
    *v10 = *(_WORD *)v8;
    *(_QWORD *)(a3 + 24) += 2LL;
  }
  else
  {
    a3 = sub_16E7EE0(a3, v8, 2u);
  }
  v11 = (__int64)v18;
  if ( v19 > 0x40 )
    v11 = *v18;
  v20[0] = v11;
  v20[1] = 0;
  v21 = v9;
  v22 = 257;
  v23 = 0;
  result = sub_16E87C0(a3, v20);
  if ( v19 > 0x40 && v18 )
    result = j_j___libc_free_0_0(v18);
  if ( v6 != v25 )
    return sub_1698460((__int64)&v25);
  v14 = v26;
  if ( v26 )
  {
    v15 = 32LL * *(_QWORD *)(v26 - 8);
    v16 = v26 + v15;
    if ( v26 != v26 + v15 )
    {
      do
      {
        v16 -= 32;
        sub_127D120((_QWORD *)(v16 + 8));
      }
      while ( v14 != v16 );
    }
    return j_j_j___libc_free_0_0(v14 - 8);
  }
  return result;
}
