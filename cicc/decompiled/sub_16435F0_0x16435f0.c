// Function: sub_16435F0
// Address: 0x16435f0
//
_BOOL8 __fastcall sub_16435F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx

  LOBYTE(v2) = *(_BYTE *)(a1 + 8);
  v3 = 35454;
  do
  {
    if ( (_BYTE)v2 != 14 && (_BYTE)v2 != 16 )
      return sub_16434A0(a1, a2);
    a1 = *(_QWORD *)(a1 + 24);
    v2 = *(unsigned __int8 *)(a1 + 8);
    if ( (unsigned __int8)v2 <= 0xFu && _bittest64(&v3, v2) )
      return 1;
  }
  while ( (unsigned int)(unsigned __int8)v2 - 13 <= 1 || (unsigned __int8)v2 == 16 );
  return 0;
}
