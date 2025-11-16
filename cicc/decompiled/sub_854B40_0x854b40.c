// Function: sub_854B40
// Address: 0x854b40
//
_QWORD *sub_854B40()
{
  _QWORD *result; // rax
  __int64 v1; // rbx
  __m128i *v2; // rdi

  result = qword_4F04C68;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v2 = *(__m128i **)(v1 + 440);
  if ( v2 )
    result = sub_854000(v2);
  *(_QWORD *)(v1 + 440) = 0;
  return result;
}
