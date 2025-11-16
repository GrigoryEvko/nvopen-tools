// Function: sub_72AC40
// Address: 0x72ac40
//
__int64 sub_72AC40()
{
  __int64 i; // rcx
  _QWORD *v1; // rsi
  __int64 result; // rax
  __int64 v3; // rdx

  for ( i = 0; i != 16312; i += 8 )
  {
    v1 = (_QWORD *)(i + qword_4F07AE0);
    for ( result = *(_QWORD *)(i + qword_4F07AE0); result; *(_QWORD *)(v3 + 120) = 0 )
    {
      v3 = result;
      result = *(_QWORD *)(result + 120);
    }
    *v1 = 0;
  }
  return result;
}
