// Function: sub_DEF800
// Address: 0xdef800
//
__int64 __fastcall sub_DEF800(__int64 a1)
{
  __int64 v3; // rsi
  __int64 v4; // rdi
  int v5; // eax
  __int64 *v6; // rdi
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+8h] [rbp-48h]
  _BYTE v11[64]; // [rsp+10h] [rbp-40h] BYREF

  if ( !*(_BYTE *)(a1 + 164) )
  {
    v3 = *(_QWORD *)(a1 + 120);
    v4 = *(_QWORD *)(a1 + 112);
    v9 = (__int64 *)v11;
    v10 = 0x400000000LL;
    v5 = sub_DBB070(v4, v3, (__int64)&v9);
    v6 = v9;
    *(_BYTE *)(a1 + 164) = 1;
    *(_DWORD *)(a1 + 160) = v5;
    v7 = &v6[(unsigned int)v10];
    if ( v7 != v6 )
    {
      v8 = v6;
      do
      {
        v3 = *v8++;
        sub_DEF380(a1, v3);
      }
      while ( v7 != v8 );
      v6 = v9;
    }
    if ( v6 != (__int64 *)v11 )
      _libc_free(v6, v3);
  }
  return *(unsigned int *)(a1 + 160);
}
