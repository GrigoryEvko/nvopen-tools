// Function: sub_AA4F90
// Address: 0xaa4f90
//
__int64 __fastcall sub_AA4F90(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  __int64 v3; // rsi

  v1 = a1 + 48;
  if ( a1 + 48 == (*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v2 = *(_QWORD *)(a1 + 56);
  if ( v1 == v2 )
    return 0;
  v3 = 0x8000018000041LL;
  while ( 1 )
  {
    if ( !v2 )
      BUG();
    if ( (unsigned __int8)(*(_BYTE *)(v2 - 24) - 34) <= 0x33u
      && _bittest64(&v3, (unsigned int)*(unsigned __int8 *)(v2 - 24) - 34) )
    {
      break;
    }
    v2 = *(_QWORD *)(v2 + 8);
    if ( v1 == v2 )
      return 0;
  }
  return v2 - 24;
}
