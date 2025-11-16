// Function: sub_155F870
// Address: 0x155f870
//
char __fastcall sub_155F870(__int64 a1, __int64 a2, void *a3, __int64 a4)
{
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 *i; // r13
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a2;
  *(_DWORD *)(a1 + 24) = a4;
  if ( 8 * a4 )
    memmove((void *)(a1 + 32), a3, 8 * a4);
  v5 = (__int64 *)sub_155EE30((__int64 *)a3);
  v6 = sub_155EE40((__int64 *)a3);
  for ( i = (__int64 *)v6; i != v5; *(_QWORD *)(a1 + 8) |= 1LL << v6 )
  {
    while ( 1 )
    {
      v9[0] = *v5;
      LOBYTE(v6) = sub_155D3E0((__int64)v9);
      if ( !(_BYTE)v6 )
        break;
      if ( i == ++v5 )
        return v6;
    }
    ++v5;
    LOBYTE(v6) = sub_155D410(v9);
  }
  return v6;
}
