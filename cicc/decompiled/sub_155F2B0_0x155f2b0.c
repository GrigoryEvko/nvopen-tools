// Function: sub_155F2B0
// Address: 0x155f2b0
//
__int64 __fastcall sub_155F2B0(__int64 a1, int a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (__int64 *)(a1 + 24);
  v3 = a1 + 8LL * *(unsigned int *)(a1 + 16) + 24;
  if ( v3 == a1 + 24 )
    return 0;
  while ( 1 )
  {
    v5[0] = *v2;
    if ( sub_155D460(v5, a2) )
      break;
    if ( (__int64 *)v3 == ++v2 )
      return 0;
  }
  return v5[0];
}
