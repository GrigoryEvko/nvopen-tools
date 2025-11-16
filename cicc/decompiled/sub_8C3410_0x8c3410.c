// Function: sub_8C3410
// Address: 0x8c3410
//
_QWORD *__fastcall sub_8C3410(__int64 a1)
{
  _QWORD *result; // rax
  __int64 i; // rbx
  _QWORD *v3; // rbx

  result = (_QWORD *)sub_8C3490(*(_QWORD *)(a1 + 104));
  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      result = (_QWORD *)sub_8C3410(*(_QWORD *)(i + 128));
  }
  if ( !*(_BYTE *)(a1 + 28) )
  {
    result = &qword_4F07280;
    v3 = (_QWORD *)qword_4F072C0;
    if ( qword_4F072C0 )
    {
      do
      {
        result = (_QWORD *)sub_8C3490(v3[3]);
        v3 = (_QWORD *)*v3;
      }
      while ( v3 );
    }
  }
  return result;
}
