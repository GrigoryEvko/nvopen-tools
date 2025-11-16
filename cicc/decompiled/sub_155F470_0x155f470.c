// Function: sub_155F470
// Address: 0x155f470
//
__int64 __fastcall sub_155F470(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r12
  __int64 v4[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = (__int64 *)(a1 + 24);
  v2 = a1 + 8LL * *(unsigned int *)(a1 + 16) + 24;
  if ( v2 == a1 + 24 )
    return 0;
  while ( 1 )
  {
    v4[0] = *v1;
    if ( sub_155D460(v4, 48) )
      break;
    if ( (__int64 *)v2 == ++v1 )
      return 0;
  }
  return sub_155D7A0(v4);
}
