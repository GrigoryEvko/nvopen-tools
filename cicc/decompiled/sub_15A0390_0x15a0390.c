// Function: sub_15A0390
// Address: 0x15a0390
//
__int64 __fastcall sub_15A0390(size_t n, __int64 a2)
{
  int v2; // r13d
  __int64 v3; // r14
  __int64 *v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // r12
  __int64 *v8; // [rsp+0h] [rbp-130h] BYREF
  __int64 v9; // [rsp+8h] [rbp-128h]
  _BYTE v10[288]; // [rsp+10h] [rbp-120h] BYREF

  v2 = n;
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 16) - 13) <= 1u && sub_15958A0(*(_QWORD *)a2) )
    return sub_15A1590((unsigned int)n);
  v3 = (unsigned int)n;
  v9 = 0x2000000000LL;
  v4 = (__int64 *)v10;
  v8 = (__int64 *)v10;
  if ( (unsigned int)n > 0x20 )
  {
    sub_16CD150(&v8, v10, (unsigned int)n, 8);
    v4 = v8;
  }
  v5 = &v4[(unsigned int)n];
  LODWORD(v9) = v2;
  if ( v4 != v5 )
  {
    do
      *v4++ = a2;
    while ( v5 != v4 );
    v5 = v8;
    v3 = (unsigned int)v9;
  }
  v6 = sub_15A01B0(v5, v3);
  if ( v8 != (__int64 *)v10 )
    _libc_free((unsigned __int64)v8);
  return v6;
}
