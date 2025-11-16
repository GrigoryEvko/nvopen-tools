// Function: sub_16D1570
// Address: 0x16d1570
//
_QWORD *__fastcall sub_16D1570(_QWORD *a1, __int64 a2, unsigned __int64 a3, const char *a4, __int64 a5)
{
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v16; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v16 = a2;
  v17 = a3;
  v8 = sub_16D24E0(&v16, a4, a5, 0);
  v9 = sub_16D23E0(&v16, a4, a5, v8);
  v10 = v17;
  if ( v9 > v17 )
  {
    v11 = v17;
    v12 = 0;
  }
  else
  {
    v11 = v9;
    v12 = v17 - v9;
  }
  v13 = v16;
  a1[3] = v12;
  if ( v8 > v10 )
    v8 = v10;
  v14 = v8 + v13;
  if ( v9 < v8 )
    v9 = v8;
  *a1 = v14;
  if ( v9 > v10 )
    v9 = v10;
  a1[1] = v9 - v8;
  a1[2] = v11 + v16;
  return a1;
}
