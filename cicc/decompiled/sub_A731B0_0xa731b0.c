// Function: sub_A731B0
// Address: 0xa731b0
//
__int64 __fastcall sub_A731B0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 *v6; // rdi
  _BYTE v8[8]; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v10[176]; // [rsp+30h] [rbp-B0h] BYREF

  v9[1] = 0x2000000000LL;
  v3 = *(_QWORD *)a1 + 64LL;
  v4 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
  v9[0] = v10;
  v5 = v3 + 8 * v4;
  while ( v5 != v3 )
  {
    v6 = (unsigned __int64 *)v3;
    v3 += 8;
    sub_A718C0(v6, (__int64)v9);
  }
  LOBYTE(v5) = *(_QWORD *)a1 == sub_C65B40(*a2 + 432LL, v9, v8, off_49D9A90);
  if ( (_BYTE *)v9[0] != v10 )
    _libc_free(v9[0], v9);
  return (unsigned int)v5;
}
