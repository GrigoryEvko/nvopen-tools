// Function: sub_8D2290
// Address: 0x8d2290
//
__int64 __fastcall sub_8D2290(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  unsigned __int64 v3; // rdx

  result = a1;
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    v2 = 6338;
    do
    {
      if ( *(_QWORD *)(result + 8) )
        break;
      v3 = *(unsigned __int8 *)(result + 184);
      if ( (unsigned __int8)v3 <= 0xCu )
      {
        if ( _bittest64(&v2, v3) )
          break;
      }
      result = *(_QWORD *)(result + 160);
    }
    while ( *(_BYTE *)(result + 140) == 12 );
  }
  return result;
}
