// Function: sub_1626700
// Address: 0x1626700
//
void __fastcall sub_1626700(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdi
  __int64 *v3; // r15
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 *v7; // [rsp+10h] [rbp-50h] BYREF
  __int64 v8; // [rsp+18h] [rbp-48h]
  _BYTE v9[64]; // [rsp+20h] [rbp-40h] BYREF

  v8 = 0x100000000LL;
  v7 = (__int64 *)v9;
  sub_1626560(a1, 0, (__int64)&v7);
  v2 = v7;
  v3 = &v7[(unsigned int)v8];
  if ( v3 != v7 )
  {
    v4 = *(unsigned int *)(a2 + 8);
    v5 = v7;
    do
    {
      v6 = *v5;
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v4 )
      {
        sub_16CD150(a2, a2 + 16, 0, 8);
        v4 = *(unsigned int *)(a2 + 8);
      }
      ++v5;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v4) = v6;
      v4 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v4;
    }
    while ( v3 != v5 );
    v2 = v7;
  }
  if ( v2 != (__int64 *)v9 )
    _libc_free((unsigned __int64)v2);
}
