// Function: sub_155F340
// Address: 0x155f340
//
__int64 __fastcall sub_155F340(__int64 a1, const void *a2, size_t a3)
{
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = (__int64 *)(a1 + 24);
  v4 = a1 + 8LL * *(unsigned int *)(a1 + 16) + 24;
  if ( v4 == a1 + 24 )
    return 0;
  while ( 1 )
  {
    v7[0] = *v3;
    if ( sub_155D850(v7, a2, a3) )
      break;
    if ( (__int64 *)v4 == ++v3 )
      return 0;
  }
  return v7[0];
}
