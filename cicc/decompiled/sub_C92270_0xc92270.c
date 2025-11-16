// Function: sub_C92270
// Address: 0xc92270
//
_QWORD *__fastcall sub_C92270(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // r8
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v18; // [rsp+8h] [rbp-38h]

  v17 = a2;
  v18 = a3;
  v8 = sub_C935B0(&v17, a4, a5, 0);
  v9 = sub_C934D0(&v17, a4, a5, v8);
  if ( v9 > v18 )
  {
    v10 = v18;
    v11 = 0;
  }
  else
  {
    v10 = v9;
    v11 = v18 - v9;
  }
  v12 = v17 + v10;
  if ( v8 > v18 )
    v8 = v18;
  v13 = 0;
  if ( v9 >= v8 )
  {
    v14 = v18;
    if ( v9 <= v18 )
      v14 = v9;
    v13 = v14 - v8;
  }
  v15 = v17 + v8;
  a1[1] = v13;
  *a1 = v15;
  a1[2] = v12;
  a1[3] = v11;
  return a1;
}
