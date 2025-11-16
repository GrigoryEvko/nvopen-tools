// Function: sub_7AEB70
// Address: 0x7aeb70
//
_BOOL8 sub_7AEB70()
{
  __m128i *v0; // rbx
  bool v1; // zf
  _BOOL8 result; // rax

  v0 = (__m128i *)qword_4F08538;
  v1 = *(_BYTE *)(qword_4F08538 + 68) == 0;
  qword_4F08560 = *(_QWORD *)(qword_4F08538 + 8);
  qword_4F08538 = *(_QWORD *)qword_4F08538;
  if ( !v1 )
    sub_7AEA70(v0 + 2);
  v0->m128i_i64[0] = qword_4F08540;
  result = 1;
  qword_4F08540 = (__int64)v0;
  if ( !unk_4D03E88 && !qword_4F08560 )
    result = qword_4F08538 != 0;
  unk_4F061FC = result;
  return result;
}
