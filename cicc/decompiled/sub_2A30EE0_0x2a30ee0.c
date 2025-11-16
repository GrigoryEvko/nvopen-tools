// Function: sub_2A30EE0
// Address: 0x2a30ee0
//
__int64 __fastcall sub_2A30EE0(_QWORD *a1)
{
  _QWORD *v1; // r14
  __int64 v2; // r13
  __int64 v3; // r15
  __m128i v4; // xmm0
  __int64 v5; // rax
  _QWORD *v6; // rbx
  __int64 v8; // [rsp+8h] [rbp-38h]

  v1 = a1;
  v2 = *a1;
  v3 = a1[1];
  v8 = a1[2];
  while ( 1 )
  {
    v5 = *(v1 - 2);
    v6 = v1;
    v1 -= 3;
    if ( (int)sub_C4C880(v2 + 24, v5 + 24) >= 0 )
      break;
    v4 = _mm_loadu_si128((const __m128i *)v1);
    v1[5] = v1[2];
    *(__m128i *)(v1 + 3) = v4;
  }
  *v6 = v2;
  v6[1] = v3;
  v6[2] = v8;
  return v8;
}
