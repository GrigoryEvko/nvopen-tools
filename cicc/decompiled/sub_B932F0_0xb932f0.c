// Function: sub_B932F0
// Address: 0xb932f0
//
unsigned __int64 __fastcall sub_B932F0(__int64 *a1)
{
  __int64 *v1; // r8
  __int64 *v2; // rdx
  __int64 *v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rdi
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v10; // [rsp+8h] [rbp-8h] BYREF

  v1 = a1 + 3;
  v2 = a1 + 2;
  v3 = a1 + 1;
  v4 = *a1;
  if ( !*a1 || *(_BYTE *)v4 != 1 )
    return sub_AF81D0(a1, v3, v2, a1 + 3);
  v5 = *(_QWORD *)(v4 + 136);
  v6 = *(__int64 **)(v5 + 24);
  v7 = *(_DWORD *)(v5 + 32);
  if ( v7 > 0x40 )
  {
    v10 = *v6;
    return sub_AF7D50(&v10, v3, v2, v1);
  }
  else
  {
    v8 = 0;
    if ( v7 )
      v8 = (__int64)((_QWORD)v6 << (64 - (unsigned __int8)v7)) >> (64 - (unsigned __int8)v7);
    v10 = v8;
    return sub_AF7D50(&v10, v3, v2, v1);
  }
}
