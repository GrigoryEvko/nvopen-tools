// Function: sub_2163260
// Address: 0x2163260
//
__int64 __fastcall sub_2163260(int a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  void *v7; // rbx
  __int64 v8; // r12
  _QWORD *v9; // rdi
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rbx
  void *v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h]

  v6 = (__int64 *)(a2 + 8);
  v7 = sub_16982C0();
  if ( *(void **)(a2 + 8) == v7 )
    sub_169C6E0(&v14, (__int64)v6);
  else
    sub_16986C0(&v14, v6);
  v8 = sub_145CBF0((__int64 *)(a3 + 48), 64, 8);
  *(_DWORD *)(v8 + 8) = 4;
  *(_QWORD *)(v8 + 16) = 0;
  v9 = (_QWORD *)(v8 + 40);
  *(_DWORD *)(v8 + 24) = a1;
  *(_QWORD *)v8 = &unk_4A01FA8;
  if ( v14 == v7 )
    sub_169C7E0(v9, &v14);
  else
    sub_1698450((__int64)v9, (__int64)&v14);
  if ( v14 == v7 )
  {
    v11 = v15;
    if ( v15 )
    {
      v12 = 32LL * *(_QWORD *)(v15 - 8);
      v13 = v15 + v12;
      if ( v15 != v15 + v12 )
      {
        do
        {
          v13 -= 32;
          sub_127D120((_QWORD *)(v13 + 8));
        }
        while ( v11 != v13 );
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v14);
  }
  return v8;
}
