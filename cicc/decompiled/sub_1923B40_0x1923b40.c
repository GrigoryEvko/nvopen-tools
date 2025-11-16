// Function: sub_1923B40
// Address: 0x1923b40
//
_QWORD *__fastcall sub_1923B40(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  char v11; // cl
  __int64 v13; // [rsp+0h] [rbp-C0h] BYREF
  char v14; // [rsp+10h] [rbp-B0h]
  __int64 v15; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v16; // [rsp+28h] [rbp-98h]
  _QWORD *v17; // [rsp+30h] [rbp-90h]
  __int64 v18; // [rsp+38h] [rbp-88h]
  int v19; // [rsp+40h] [rbp-80h]
  _QWORD v20[8]; // [rsp+48h] [rbp-78h] BYREF
  unsigned __int64 v21; // [rsp+88h] [rbp-38h] BYREF
  __int64 v22; // [rsp+90h] [rbp-30h]
  __int64 v23; // [rsp+98h] [rbp-28h]

  v16 = v20;
  v17 = v20;
  v20[0] = a2;
  v13 = a2;
  v18 = 0x100000008LL;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v19 = 0;
  v15 = 1;
  v14 = 0;
  sub_1923AF0(&v21, (__int64)&v13);
  sub_16CCCB0(a1, (__int64)(a1 + 5), (__int64)&v15);
  v4 = v22;
  v5 = v21;
  a1[13] = 0;
  a1[14] = 0;
  a1[15] = 0;
  v6 = v4 - v5;
  if ( v4 == v5 )
  {
    v8 = 0;
  }
  else
  {
    if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, v4, v3);
    v7 = sub_22077B0(v4 - v5);
    v4 = v22;
    v5 = v21;
    v8 = v7;
  }
  a1[13] = v8;
  a1[14] = v8;
  a1[15] = v8 + v6;
  if ( v4 != v5 )
  {
    v9 = v8;
    v10 = v5;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v10;
        v11 = *(_BYTE *)(v10 + 16);
        *(_BYTE *)(v9 + 16) = v11;
        if ( v11 )
          *(_QWORD *)(v9 + 8) = *(_QWORD *)(v10 + 8);
      }
      v10 += 24LL;
      v9 += 24;
    }
    while ( v10 != v4 );
    v8 += 8 * ((v10 - 24 - v5) >> 3) + 24;
  }
  a1[14] = v8;
  if ( v5 )
    j_j___libc_free_0(v5, v23 - v5);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
  return a1;
}
