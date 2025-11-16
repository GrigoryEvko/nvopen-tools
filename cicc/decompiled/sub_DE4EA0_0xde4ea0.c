// Function: sub_DE4EA0
// Address: 0xde4ea0
//
__int64 __fastcall sub_DE4EA0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v7; // [rsp+0h] [rbp-80h] BYREF
  char *v8; // [rsp+8h] [rbp-78h]
  __int64 v9; // [rsp+10h] [rbp-70h]
  int v10; // [rsp+18h] [rbp-68h]
  char v11; // [rsp+1Ch] [rbp-64h]
  char v12; // [rsp+20h] [rbp-60h] BYREF

  v4 = **(_QWORD **)(a2 + 32);
  v5 = sub_D47840(a2);
  *(_QWORD *)(a1 + 40) = a3;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_WORD *)(a1 + 32) = 0;
  if ( !v5 )
    return a1;
  v8 = &v12;
  v7 = 0;
  v9 = 8;
  v10 = 0;
  v11 = 1;
  sub_DE2750(a3, a1, v4, v5, (__int64)&v7, 0);
  if ( v11 )
    return a1;
  _libc_free(v8, a1);
  return a1;
}
