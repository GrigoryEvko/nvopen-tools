// Function: sub_1F39810
// Address: 0x1f39810
//
char *__fastcall sub_1F39810(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  __int64 *i; // rsi
  __int64 v7; // rsi
  _QWORD *v8; // r14
  unsigned __int64 *v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  char *result; // rax
  __int64 v14; // [rsp+18h] [rbp-48h] BYREF
  _BYTE *v15; // [rsp+20h] [rbp-40h]
  __int64 v16; // [rsp+28h] [rbp-38h]
  _BYTE v17[48]; // [rsp+30h] [rbp-30h] BYREF

  v4 = a2;
  v5 = (_QWORD *)a2[3];
  for ( i = (__int64 *)v5[11]; i != (__int64 *)v5[12]; i = (__int64 *)v5[11] )
    sub_1DD9130((__int64)v5, i, 0);
  v7 = a2[8];
  v14 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v14, v7, 2);
  if ( a2 != v5 + 3 )
  {
    do
    {
      v8 = v4;
      v4 = (_QWORD *)v4[1];
      sub_1DD5BC0((__int64)(v5 + 2), (__int64)v8);
      v9 = (unsigned __int64 *)v8[1];
      v10 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      *v9 = v10 | *v9 & 7;
      *(_QWORD *)(v10 + 8) = v9;
      *v8 &= 7uLL;
      v8[1] = 0;
      sub_1DD5C20((__int64)(v5 + 2));
    }
    while ( v5 + 3 != v4 );
  }
  if ( a3 != v5[1] )
  {
    v15 = v17;
    v11 = *a1;
    v16 = 0;
    (*(void (__fastcall **)(__int64 *, _QWORD *, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(v11 + 288))(
      a1,
      v5,
      a3,
      0,
      v17,
      0,
      &v14,
      0);
    if ( v15 != v17 )
      _libc_free((unsigned __int64)v15);
  }
  result = sub_1DD8FE0((__int64)v5, a3, -1);
  if ( v14 )
    return (char *)sub_161E7C0((__int64)&v14, v14);
  return result;
}
