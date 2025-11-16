// Function: sub_7252B0
// Address: 0x7252b0
//
__int64 *__fastcall sub_7252B0(__int64 a1)
{
  __int64 *result; // rax
  __int64 *v2; // rdx

  result = (__int64 *)qword_4F07968;
  if ( qword_4F07968 )
  {
    do
    {
      v2 = result;
      result = (__int64 *)*result;
    }
    while ( result );
    *v2 = a1;
  }
  else
  {
    qword_4F07968 = a1;
  }
  return result;
}
