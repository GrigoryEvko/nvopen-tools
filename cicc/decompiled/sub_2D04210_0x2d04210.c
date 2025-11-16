// Function: sub_2D04210
// Address: 0x2d04210
//
__int64 __fastcall sub_2D04210(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx

  result = 0;
  if ( *(_BYTE *)a1 == 22 )
  {
    v2 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL);
    if ( (unsigned __int8)v2 <= 3u )
      return (unsigned int)sub_B2D670(a1, 81) ^ 1;
    if ( (_BYTE)v2 == 5 )
      return (unsigned int)sub_B2D670(a1, 81) ^ 1;
    if ( (unsigned __int8)v2 <= 0x14u )
    {
      v3 = 1463376;
      if ( _bittest64(&v3, v2) )
        return (unsigned int)sub_B2D670(a1, 81) ^ 1;
    }
  }
  return result;
}
