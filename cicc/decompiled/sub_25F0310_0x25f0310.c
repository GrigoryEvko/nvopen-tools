// Function: sub_25F0310
// Address: 0x25f0310
//
bool __fastcall sub_25F0310(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  __int64 i; // rax

  if ( (*(_WORD *)(a1 + 2) & 0x7FFF) != 0 )
    return 0;
  v2 = sub_AA4FF0(a1);
  if ( !v2 )
    BUG();
  v3 = (unsigned int)*(unsigned __int8 *)(v2 - 24) - 39;
  if ( (unsigned int)v3 <= 0x38 )
  {
    v4 = 0x100060000000001LL;
    if ( _bittest64(&v4, v3) )
      return 0;
  }
  v5 = a1 + 48;
  v6 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1 + 48 == v6 )
    goto LABEL_18;
  if ( !v6 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
LABEL_18:
    BUG();
  if ( (unsigned __int8)(*(_BYTE *)(v6 - 24) - 34) <= 1u )
    return 0;
  for ( i = *(_QWORD *)(a1 + 56); v5 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) == 11 )
      break;
  }
  return v5 == i;
}
