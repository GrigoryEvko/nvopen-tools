// Function: sub_242E160
// Address: 0x242e160
//
unsigned __int64 *__fastcall sub_242E160(unsigned __int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 *v4; // r12
  __int64 v5; // rdx
  unsigned __int64 v6; // rbx
  __int64 v7; // r9
  unsigned __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-140h]
  __int64 v15; // [rsp+18h] [rbp-138h]
  __int64 v17; // [rsp+30h] [rbp-120h] BYREF
  unsigned __int64 v18; // [rsp+38h] [rbp-118h]
  __int64 v19[6]; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int64 v20[2]; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE v21[16]; // [rsp+80h] [rbp-D0h] BYREF
  _QWORD v22[4]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v23; // [rsp+B0h] [rbp-A0h]
  _QWORD v24[4]; // [rsp+C0h] [rbp-90h] BYREF
  __int16 v25; // [rsp+E0h] [rbp-70h]
  _QWORD v26[4]; // [rsp+F0h] [rbp-60h] BYREF
  __int16 v27; // [rsp+110h] [rbp-40h]

  v4 = a1;
  v18 = a4;
  v17 = a3;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a4 )
  {
    while ( 1 )
    {
      LOBYTE(v26[0]) = 59;
      v10 = sub_C931B0(&v17, v26, 1u, 0);
      if ( v10 != -1 )
        break;
      v5 = v18;
      v7 = v17;
      v8 = 0;
      v9 = 0;
      if ( v18 )
        goto LABEL_11;
LABEL_8:
      v17 = v9;
      v18 = v8;
      if ( !v8 )
        return a1;
    }
    v5 = v18;
    v6 = v10 + 1;
    v7 = v17;
    if ( v10 + 1 > v18 )
    {
      v6 = v18;
      v8 = 0;
    }
    else
    {
      v8 = v18 - v6;
    }
    v9 = v17 + v6;
    if ( v10 <= v18 )
      v5 = v10;
    if ( !v5 )
      goto LABEL_8;
LABEL_11:
    v14 = v7;
    v15 = v5;
    sub_C88F40((__int64)v19, v7, v5, 0);
    v20[0] = (unsigned __int64)v21;
    v20[1] = 0;
    v21[0] = 0;
    if ( !(unsigned __int8)sub_C89030(v19, v20) )
    {
      v22[3] = v15;
      v22[2] = v14;
      v13 = *(_QWORD *)(a2 + 168);
      v25 = 770;
      v22[0] = "Regex ";
      v23 = 1283;
      v24[0] = v22;
      v24[2] = " is not valid: ";
      v26[0] = v24;
      v27 = 1026;
      v26[2] = v20;
      sub_B6ECE0(v13, (__int64)v26);
    }
    v11 = a1[1];
    if ( v11 == a1[2] )
    {
      sub_242DFA0(a1, (__int64 *)a1[1], v19);
    }
    else
    {
      if ( v11 )
      {
        sub_C88FD0(v11, v19);
        v11 = a1[1];
      }
      a1[1] = v11 + 16;
    }
    if ( (_BYTE *)v20[0] != v21 )
      j_j___libc_free_0(v20[0]);
    sub_C88FF0(v19);
    goto LABEL_8;
  }
  return v4;
}
