// Function: sub_854AB0
// Address: 0x854ab0
//
_QWORD *sub_854AB0()
{
  _QWORD *result; // rax
  __int64 v1; // r12
  __m128i *v2; // r13
  __int64 *v3; // rbx
  unsigned __int8 v4; // di

  result = qword_4F04C68;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v2 = *(__m128i **)(v1 + 440);
  if ( v2 )
  {
    v3 = *(__int64 **)(v1 + 440);
    do
    {
      v4 = *(_BYTE *)(v3[1] + 19);
      if ( v4 != 3 )
        sub_684AA0(v4, 0x261u, (_DWORD *)v3 + 12);
      v3 = (__int64 *)*v3;
    }
    while ( v3 );
    result = sub_854000(v2);
  }
  *(_QWORD *)(v1 + 440) = 0;
  return result;
}
