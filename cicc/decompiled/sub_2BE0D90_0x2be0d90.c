// Function: sub_2BE0D90
// Address: 0x2be0d90
//
void __fastcall sub_2BE0D90(__int64 a1, _QWORD *a2, const __m128i **a3)
{
  _QWORD *v5; // r12
  unsigned __int64 v6; // rbx
  __int64 v7; // rdi
  const __m128i *v8; // rcx
  const __m128i *v9; // r8
  __m128i *v10; // rdx
  const __m128i *v11; // rax

  v5 = *(_QWORD **)(a1 + 8);
  if ( v5 == *(_QWORD **)(a1 + 16) )
  {
    sub_2BE0AC0((unsigned __int64 *)a1, *(char **)(a1 + 8), a2, a3);
  }
  else
  {
    if ( v5 )
    {
      *v5 = *a2;
      v6 = (char *)a3[1] - (char *)*a3;
      v5[1] = 0;
      v5[2] = 0;
      v5[3] = 0;
      if ( v6 )
      {
        if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
          sub_4261EA(a1, a2, a3);
        v7 = sub_22077B0(v6);
      }
      else
      {
        v6 = 0;
        v7 = 0;
      }
      v5[1] = v7;
      v5[2] = v7;
      v5[3] = v7 + v6;
      v8 = a3[1];
      v9 = *a3;
      if ( v8 != *a3 )
      {
        v10 = (__m128i *)v7;
        v11 = *a3;
        do
        {
          if ( v10 )
          {
            *v10 = _mm_loadu_si128(v11);
            v10[1].m128i_i64[0] = v11[1].m128i_i64[0];
          }
          v11 = (const __m128i *)((char *)v11 + 24);
          v10 = (__m128i *)((char *)v10 + 24);
        }
        while ( v8 != v11 );
        v7 += 8 * ((unsigned __int64)((char *)&v8[-2].m128i_u64[1] - (char *)v9) >> 3) + 24;
      }
      v5[2] = v7;
      v5 = *(_QWORD **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v5 + 4;
  }
}
