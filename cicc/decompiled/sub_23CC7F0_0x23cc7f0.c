// Function: sub_23CC7F0
// Address: 0x23cc7f0
//
void __fastcall sub_23CC7F0(pthread_t *a1, __int64 a2, const __m128i *a3)
{
  __m128i *v4; // rax
  unsigned __int64 v5; // r12
  pthread_t v6; // rax

  v4 = (__m128i *)sub_22077B0(0x10u);
  if ( v4 )
  {
    v5 = (unsigned __int64)v4;
    *v4 = _mm_loadu_si128(a3);
    v6 = sub_C958F0(sub_23CE0C0, v4, a2);
    *a1 = v6;
    if ( !v6 )
      j_j___libc_free_0(v5);
  }
  else
  {
    *a1 = sub_C958F0(sub_23CE0C0, 0, a2);
  }
}
