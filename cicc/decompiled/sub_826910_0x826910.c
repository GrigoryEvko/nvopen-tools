// Function: sub_826910
// Address: 0x826910
//
_QWORD *__fastcall sub_826910(int a1, unsigned int a2)
{
  _QWORD *result; // rax

  result = &unk_4F1F660;
  if ( !a1 )
    result = &unk_4F1F6E0;
  result[a2 >> 6] |= 1LL << a2;
  return result;
}
