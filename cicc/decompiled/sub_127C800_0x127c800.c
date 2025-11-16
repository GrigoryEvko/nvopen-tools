// Function: sub_127C800
// Address: 0x127c800
//
__int64 __fastcall sub_127C800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(a1 + 156) & 2) == 0
    || *(_QWORD *)(i + 128)
    || !sub_8D3410(i)
    || *(_QWORD *)(i + 176)
    || (*(_BYTE *)(i + 169) & 0x20) != 0
    || *(_BYTE *)(a1 + 136) != 1 )
  {
    return sub_736C20(a1, a2, a3);
  }
  result = sub_736C20(a1, a2, a3);
  if ( !HIDWORD(qword_4D045BC) && (unsigned int)result < 0x10 )
    return 16;
  return result;
}
