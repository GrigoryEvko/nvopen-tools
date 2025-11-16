// Function: sub_1704BC0
// Address: 0x1704bc0
//
_BOOL8 __fastcall sub_1704BC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rdx

  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned __int8)v2 <= 0xFu )
  {
    v4 = 35454;
    if ( _bittest64(&v4, v2) )
      return 1;
  }
  if ( (unsigned int)(v2 - 13) <= 1 || (_DWORD)v2 == 16 )
    return sub_16435F0(a1, a2);
  return 0;
}
