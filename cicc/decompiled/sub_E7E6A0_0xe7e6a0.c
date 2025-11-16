// Function: sub_E7E6A0
// Address: 0xe7e6a0
//
unsigned __int64 __fastcall sub_E7E6A0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax

  while ( 1 )
  {
    while ( 1 )
    {
      result = *a2;
      if ( (_BYTE)result != 3 )
        break;
      a2 = (unsigned __int8 *)*((_QWORD *)a2 + 2);
    }
    if ( (unsigned __int8)result > 3u )
    {
      if ( (_BYTE)result == 4 )
        return (*(__int64 (__fastcall **)(unsigned __int8 *, _QWORD))(*((_QWORD *)a2 - 1) + 80LL))(
                 a2 - 8,
                 *(_QWORD *)(a1 + 296));
      return result;
    }
    if ( (_BYTE)result )
      break;
    sub_E7E6A0(a1, *((_QWORD *)a2 + 2));
    a2 = (unsigned __int8 *)*((_QWORD *)a2 + 3);
  }
  if ( (_BYTE)result == 2 )
  {
    result = *(_DWORD *)a2 >> 8;
    if ( (unsigned __int16)result <= 0x9Du )
    {
      if ( (unsigned __int16)result > 0x65u )
      {
        a3 = 0xC00000000001AFLL;
        result = (unsigned int)(result - 102);
        if ( !_bittest64(&a3, result) )
          return result;
      }
      else if ( (unsigned __int16)result > 0x13u )
      {
        result = (unsigned int)(result - 64);
        if ( (unsigned __int16)result > 0x1Fu )
          return result;
      }
      else if ( (unsigned __int16)result <= 0xCu )
      {
        result = (unsigned int)(result - 8);
        if ( (unsigned __int16)result > 3u )
          return result;
      }
      sub_E5CB20(*(_QWORD *)(a1 + 296), *((_QWORD *)a2 + 2), a3, a4, a5, a6);
      return sub_EA15B0(*((_QWORD *)a2 + 2), 6);
    }
  }
  return result;
}
