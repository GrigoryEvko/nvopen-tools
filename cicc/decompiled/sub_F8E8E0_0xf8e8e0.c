// Function: sub_F8E8E0
// Address: 0xf8e8e0
//
__int64 __fastcall sub_F8E8E0(__m128i **a1, __int64 a2)
{
  __m128i *v2; // r12
  __m128i *v3; // rcx
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rdx
  __m128i *v7; // rdx
  __m128i *v8; // rax
  __m128i *v9; // rcx

  v2 = a1[1];
  v3 = *a1;
  v4 = (char *)v2 - (char *)*a1;
  result = v4 >> 6;
  v6 = v4 >> 4;
  if ( result > 0 )
  {
    result = (__int64)v3[4 * result].m128i_i64;
    while ( v3->m128i_i64[1] != a2 )
    {
      if ( v3[1].m128i_i64[1] == a2 )
      {
        ++v3;
        goto LABEL_8;
      }
      if ( v3[2].m128i_i64[1] == a2 )
      {
        v3 += 2;
        goto LABEL_8;
      }
      if ( v3[3].m128i_i64[1] == a2 )
      {
        v3 += 3;
        goto LABEL_8;
      }
      v3 += 4;
      if ( (__m128i *)result == v3 )
      {
        v6 = v2 - v3;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
LABEL_20:
  if ( v6 == 2 )
  {
LABEL_27:
    if ( v3->m128i_i64[1] != a2 )
    {
      ++v3;
      goto LABEL_23;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return result;
LABEL_23:
    if ( v3->m128i_i64[1] != a2 )
      return result;
    goto LABEL_8;
  }
  if ( v3->m128i_i64[1] != a2 )
  {
    ++v3;
    goto LABEL_27;
  }
LABEL_8:
  if ( v2 != v3 )
  {
    result = (__int64)v3[1].m128i_i64;
    if ( v2 == &v3[1] )
      goto LABEL_14;
    do
    {
      if ( *(_QWORD *)(result + 8) != a2 )
      {
        ++v3;
        v3[-1] = _mm_loadu_si128((const __m128i *)result);
      }
      result += 16;
    }
    while ( v2 != (__m128i *)result );
    if ( v2 != v3 )
    {
LABEL_14:
      v7 = a1[1];
      if ( v2 != v7 )
      {
        v8 = (__m128i *)memmove(v3, v2, (char *)v7 - (char *)v2);
        v7 = a1[1];
        v3 = v8;
      }
      result = (char *)v7 - (char *)v2;
      v9 = (__m128i *)((char *)v3 + (char *)v7 - (char *)v2);
      if ( v9 != v7 )
        a1[1] = v9;
    }
  }
  return result;
}
