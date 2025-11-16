// Function: sub_A40FA0
// Address: 0xa40fa0
//
__m128i *__fastcall sub_A40FA0(__m128i *a1, __int64 *a2, unsigned __int8 (__fastcall *a3)(char *))
{
  __int64 v4; // r14
  __m128i *v5; // r12
  __int64 v6; // rax
  __m128i *v7; // r14
  __m128i *v8; // rbx
  __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]

  v4 = ((char *)a2 - (char *)a1) >> 6;
  v5 = a1;
  v6 = ((char *)a2 - (char *)a1) >> 4;
  if ( v4 > 0 )
  {
    v7 = &a1[4 * v4];
    while ( a3(v5->m128i_i8) )
    {
      v8 = v5++;
      if ( !a3(v5->m128i_i8) )
        break;
      v5 = v8 + 2;
      if ( !a3(v8[2].m128i_i8) )
        break;
      v5 = v8 + 3;
      if ( !a3(v8[3].m128i_i8) )
        break;
      v5 = v8 + 4;
      if ( &v8[4] == v7 )
      {
        v6 = ((char *)a2 - (char *)v5) >> 4;
        goto LABEL_12;
      }
    }
    goto LABEL_8;
  }
LABEL_12:
  if ( v6 != 2 )
  {
    if ( v6 != 3 )
    {
      if ( v6 != 1 )
        return (__m128i *)a2;
      goto LABEL_20;
    }
    if ( !a3(v5->m128i_i8) )
      goto LABEL_8;
    ++v5;
  }
  if ( !a3(v5->m128i_i8) )
    goto LABEL_8;
  ++v5;
LABEL_20:
  if ( a3(v5->m128i_i8) )
    return (__m128i *)a2;
LABEL_8:
  if ( a2 != (__int64 *)v5 )
  {
    sub_A40D50(&v10, v5, ((char *)a2 - (char *)v5) >> 4);
    v5 = (__m128i *)sub_A40E10(v5->m128i_i8, a2, a3, v10, v12, v11);
    j_j___libc_free_0(v12, 16 * v11);
  }
  return v5;
}
