// Function: sub_7F6D70
// Address: 0x7f6d70
//
__int64 __fastcall sub_7F6D70(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  _QWORD *i; // rbx

  result = (unsigned int)*(unsigned __int8 *)(a1 + 174) - 1;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u && *(char *)(a1 + 197) >= 0 && a2 )
  {
    if ( (a2[156] & 1) != 0 )
    {
      if ( (*(_BYTE *)(a1 + 198) & 0x10) == 0 )
      {
        result = ((__int64 (*)(void))sub_825750)();
        *(_BYTE *)(a1 + 198) |= 0x10u;
        if ( !HIDWORD(qword_4D045BC) )
          *(_BYTE *)(a1 + 192) |= 0x80u;
        for ( i = *(_QWORD **)(a1 + 176); i; i = (_QWORD *)*i )
        {
          sub_825750(i[1]);
          *(_BYTE *)(i[1] + 198LL) |= 0x10u;
          result = HIDWORD(qword_4D045BC);
          if ( !HIDWORD(qword_4D045BC) )
          {
            result = i[1];
            *(_BYTE *)(result + 192) |= 0x80u;
          }
        }
      }
    }
    else if ( a2[136] <= 2u && (a2[174] & 4) == 0 )
    {
      result = *(unsigned __int8 *)(a1 + 198);
      if ( (*(_BYTE *)(a1 + 198) & 0x18) == 0x10 )
      {
        *(_BYTE *)(a1 + 198) = result | 8;
        for ( result = *(_QWORD *)(a1 + 176); result; result = *(_QWORD *)result )
          *(_BYTE *)(*(_QWORD *)(result + 8) + 198LL) |= 8u;
      }
    }
  }
  return result;
}
