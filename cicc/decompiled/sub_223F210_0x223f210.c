// Function: sub_223F210
// Address: 0x223f210
//
__int64 __fastcall sub_223F210(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int8 *v3; // rax

  if ( (*(_BYTE *)(a1 + 64) & 8) == 0 )
    return 0xFFFFFFFFLL;
  v1 = *(_QWORD *)(a1 + 40);
  v2 = *(_QWORD *)(a1 + 24);
  if ( v1 )
  {
    if ( v1 > v2 )
    {
      *(_QWORD *)(a1 + 24) = v1;
      v2 = v1;
    }
  }
  v3 = *(unsigned __int8 **)(a1 + 16);
  if ( (unsigned __int64)v3 < v2 )
    return *v3;
  else
    return 0xFFFFFFFFLL;
}
