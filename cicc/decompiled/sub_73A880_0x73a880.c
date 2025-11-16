// Function: sub_73A880
// Address: 0x73a880
//
_QWORD *__fastcall sub_73A880(const __m128i *a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdi
  _QWORD *v4; // r12

  if ( a2 )
  {
    v2 = sub_730690(a2);
    v3 = *(_QWORD *)(a2 + 128);
  }
  else
  {
    v2 = sub_73A720(a1, 0);
    v3 = a1[8].m128i_i64[0];
  }
  v4 = v2;
  if ( (unsigned int)sub_8D32E0(v3) )
    *((_BYTE *)v4 + 25) |= 1u;
  return v4;
}
