// Function: sub_299F8F0
// Address: 0x299f8f0
//
void __fastcall sub_299F8F0(__int64 a1, unsigned __int8 (__fastcall *a2)(__m128i *, __int8 *))
{
  __m128i *v2; // r14
  __int64 v3; // rax
  __m128i *v4; // r15
  __int64 v5; // rcx
  __m128i *v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // [rsp+8h] [rbp-38h]

  v2 = *(__m128i **)a1;
  v3 = 56LL * *(unsigned int *)(a1 + 8);
  v4 = (__m128i *)(*(_QWORD *)a1 + v3);
  v5 = 0x6DB6DB6DB6DB6DB7LL * (v3 >> 3);
  if ( v3 )
  {
    while ( 1 )
    {
      v8 = v5;
      v6 = (__m128i *)sub_2207800(56 * v5);
      v7 = (unsigned __int64)v6;
      if ( v6 )
        break;
      v5 = v8 >> 1;
      if ( !(v8 >> 1) )
        goto LABEL_5;
    }
    sub_299F7D0(v2, v4, v6, v8, a2);
    j_j___libc_free_0(v7);
  }
  else
  {
LABEL_5:
    sub_299EE90(v2->m128i_i8, v4->m128i_i8, a2);
    j_j___libc_free_0(0);
  }
}
