// Function: sub_2CF5CA0
// Address: 0x2cf5ca0
//
bool __fastcall sub_2CF5CA0(__int64 a1)
{
  unsigned __int8 *v1; // rbx
  bool result; // al
  unsigned __int64 v3; // rax
  __int64 v4; // rdx

  v1 = *(unsigned __int8 **)(a1 - 32);
  result = sub_BCAC40(*((_QWORD *)v1 + 1), 32);
  if ( result )
  {
    v3 = *v1;
    if ( (unsigned __int8)v3 <= 0x1Cu )
    {
      if ( (_BYTE)v3 == 5 )
      {
        result = (*((_WORD *)v1 + 1) & 0xFFFD) == 13 || (*((_WORD *)v1 + 1) & 0xFFF7) == 17;
        if ( !result )
          return result;
        return (v1[1] & 4) != 0;
      }
    }
    else if ( (unsigned __int8)v3 <= 0x36u )
    {
      v4 = 0x40540000000000LL;
      if ( _bittest64(&v4, v3) )
        return (v1[1] & 4) != 0;
    }
    return (_BYTE)v3 == 61;
  }
  return result;
}
