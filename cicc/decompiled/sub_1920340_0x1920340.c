// Function: sub_1920340
// Address: 0x1920340
//
_QWORD *__fastcall sub_1920340(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  char v10; // si
  _QWORD v12[18]; // [rsp+0h] [rbp-90h] BYREF

  memset(v12, 0, 0x80u);
  LODWORD(v12[3]) = 8;
  v12[1] = &v12[5];
  v12[2] = &v12[5];
  sub_16CCCB0(a1, (__int64)(a1 + 5), (__int64)v12);
  v3 = v12[14];
  v4 = v12[13];
  a1[13] = 0;
  a1[14] = 0;
  a1[15] = 0;
  v5 = v3 - v4;
  if ( v3 == v4 )
  {
    v7 = 0;
  }
  else
  {
    if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a1 + 5, v2);
    v6 = sub_22077B0(v3 - v4);
    v3 = v12[14];
    v4 = v12[13];
    v7 = v6;
  }
  a1[13] = v7;
  a1[14] = v7;
  a1[15] = v7 + v5;
  if ( v3 != v4 )
  {
    v8 = v7;
    v9 = v4;
    do
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = *(_QWORD *)v9;
        v10 = *(_BYTE *)(v9 + 16);
        *(_BYTE *)(v8 + 16) = v10;
        if ( v10 )
          *(_QWORD *)(v8 + 8) = *(_QWORD *)(v9 + 8);
      }
      v9 += 24;
      v8 += 24;
    }
    while ( v3 != v9 );
    v7 += 8 * ((unsigned __int64)(v3 - 24 - v4) >> 3) + 24;
  }
  a1[14] = v7;
  if ( v4 )
    j_j___libc_free_0(v4, v12[15] - v4);
  if ( v12[2] != v12[1] )
    _libc_free(v12[2]);
  return a1;
}
