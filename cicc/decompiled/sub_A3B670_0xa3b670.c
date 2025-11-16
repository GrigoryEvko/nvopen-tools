// Function: sub_A3B670
// Address: 0xa3b670
//
void __fastcall sub_A3B670(__m128i *src, __m128i *a2)
{
  __m128i *v2; // rbx
  __m128i *v3; // r14
  unsigned __int64 v4; // r15
  size_t v5; // r14
  __int64 v6; // r13
  size_t v7; // rdx
  int v8; // eax

  if ( src != a2 )
  {
    v2 = src + 1;
    if ( &src[1] != a2 )
    {
      while ( 1 )
      {
        v4 = v2->m128i_u64[1];
        v5 = src->m128i_u64[1];
        v6 = v2->m128i_i64[0];
        v7 = v5;
        if ( v4 <= v5 )
          v7 = v2->m128i_u64[1];
        if ( v7 && (v8 = memcmp((const void *)v2->m128i_i64[0], (const void *)src->m128i_i64[0], v7)) != 0 )
        {
          if ( v8 < 0 )
            goto LABEL_4;
LABEL_13:
          sub_A3B600(v2++);
          if ( a2 == v2 )
            return;
        }
        else
        {
          if ( v4 == v5 || v4 >= v5 )
            goto LABEL_13;
LABEL_4:
          v3 = v2 + 1;
          if ( src != v2 )
            memmove(&src[1], src, (char *)v2 - (char *)src);
          src->m128i_i64[0] = v6;
          ++v2;
          src->m128i_i64[1] = v4;
          if ( a2 == v3 )
            return;
        }
      }
    }
  }
}
