// Function: sub_DD92B0
// Address: 0xdd92b0
//
__int64 *__fastcall sub_DD92B0(__int64 *a1, __int64 a2)
{
  __int64 *i; // rbx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 *v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 *v8; // r12
  _BYTE *v10; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  _BYTE v12[80]; // [rsp+20h] [rbp-50h] BYREF

  v10 = v12;
  v11 = 0x400000000LL;
  for ( i = (__int64 *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))); (__int64 *)a2 != i; LODWORD(v11) = v11 + 1 )
  {
    v5 = sub_DD8400((__int64)a1, *i);
    v6 = (unsigned int)v11;
    v7 = (unsigned int)v11 + 1LL;
    if ( v7 > HIDWORD(v11) )
    {
      sub_C8D5F0((__int64)&v10, v12, v7, 8u, v3, v4);
      v6 = (unsigned int)v11;
    }
    i += 4;
    *(_QWORD *)&v10[8 * v6] = v5;
  }
  v8 = sub_DD8EB0(a1, a2, (__int64)&v10);
  if ( v10 != v12 )
    _libc_free(v10, a2);
  return v8;
}
