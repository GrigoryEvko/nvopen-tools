// Function: sub_7DC070
// Address: 0x7dc070
//
__int64 sub_7DC070()
{
  __int64 result; // rax
  const __m128i *v1; // rax
  __m128i *v2; // rax

  result = qword_4F18940;
  if ( !qword_4F18940 )
  {
    v1 = (const __m128i *)sub_7DB910(0, 0);
    v2 = sub_73C570(v1, 1);
    result = sub_72D2E0(v2);
    qword_4F18940 = result;
  }
  return result;
}
