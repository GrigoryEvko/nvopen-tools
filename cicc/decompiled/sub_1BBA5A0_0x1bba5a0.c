// Function: sub_1BBA5A0
// Address: 0x1bba5a0
//
__int64 __fastcall sub_1BBA5A0(unsigned int a1, int a2, unsigned __int8 a3, __int64 a4)
{
  __int64 **v6; // rax
  __int64 v7; // rbx
  int v8; // r8d
  int v9; // r9d
  __int64 *v10; // rax
  __int64 *v11; // rdi
  unsigned __int64 v12; // r13
  __int64 v13; // r14
  unsigned int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r12
  __int64 *v20; // [rsp+10h] [rbp-140h] BYREF
  __int64 v21; // [rsp+18h] [rbp-138h]
  _BYTE v22[304]; // [rsp+20h] [rbp-130h] BYREF

  v6 = (__int64 **)sub_1643350(*(_QWORD **)(a4 + 24));
  v7 = sub_1599EF0(v6);
  v10 = (__int64 *)v22;
  v21 = 0x2000000000LL;
  v20 = (__int64 *)v22;
  if ( a1 > 0x20 )
  {
    sub_16CD150((__int64)&v20, v22, a1, 8, v8, v9);
    v10 = v20;
  }
  v11 = &v10[a1];
  LODWORD(v21) = a1;
  if ( v10 != v11 )
  {
    do
      *v10++ = v7;
    while ( v11 != v10 );
    v11 = v20;
  }
  if ( a2 )
  {
    v12 = 0;
    v13 = 8LL * (unsigned int)(a2 - 1);
    v14 = a3 ^ 1;
    while ( 1 )
    {
      v15 = sub_1643350(*(_QWORD **)(a4 + 24));
      v16 = v14;
      v14 += 2;
      v11[v12 / 8] = sub_159C470(v15, v16, 0);
      v11 = v20;
      if ( v13 == v12 )
        break;
      v12 += 8LL;
    }
  }
  v17 = sub_15A01B0(v11, (unsigned int)v21);
  if ( v20 != (__int64 *)v22 )
    _libc_free((unsigned __int64)v20);
  return v17;
}
