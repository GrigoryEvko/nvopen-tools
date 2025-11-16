// Function: sub_22AE6A0
// Address: 0x22ae6a0
//
__int64 __fastcall sub_22AE6A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  __int64 result; // rax
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v3 = *(_DWORD *)(v2 + 36);
  if ( v3 <= 0xD3 )
  {
    if ( v3 <= 0x9A )
    {
      if ( v3 == 11 || v3 - 68 <= 3 )
        return 1;
      return *(_BYTE *)(a1 + 2) ^ 1u;
    }
    v5 = 0x186000000000001LL;
    if ( !_bittest64(&v5, v3 - 155) )
      return *(_BYTE *)(a1 + 2) ^ 1u;
    return 1;
  }
  if ( v3 == 324 )
    return 1;
  result = 1;
  if ( v3 <= 0x144 )
  {
    if ( v3 != 282 && v3 - 291 > 1 )
      return *(_BYTE *)(a1 + 2) ^ 1u;
  }
  else if ( v3 != 376 )
  {
    return *(_BYTE *)(a1 + 2) ^ 1u;
  }
  return result;
}
