// Function: sub_19E27F0
// Address: 0x19e27f0
//
__int64 __fastcall sub_19E27F0(__m128i *src, __m128i *a2)
{
  __m128i *v2; // r9
  __int32 v3; // r12d
  __int32 v4; // ecx
  __int32 v5; // r8d
  __int64 v6; // r10
  __m128i *v7; // r15
  __int64 result; // rax
  unsigned __int64 v9; // r14
  __int64 v10; // [rsp-48h] [rbp-48h]
  __int32 v11; // [rsp-40h] [rbp-40h]
  __int32 v12; // [rsp-3Ch] [rbp-3Ch]

  if ( src != a2 )
  {
    v2 = src + 2;
    if ( a2 != &src[2] )
    {
      while ( 1 )
      {
        v3 = v2->m128i_i32[0];
        if ( v2->m128i_i32[0] < src->m128i_i32[0] )
          break;
        if ( v3 == src->m128i_i32[0] )
        {
          v4 = v2->m128i_i32[1];
          if ( v4 < src->m128i_i32[1] )
            goto LABEL_15;
          if ( v4 == src->m128i_i32[1] )
          {
            v5 = v2->m128i_i32[2];
            if ( v5 < src->m128i_i32[2] )
              goto LABEL_16;
            if ( v5 == src->m128i_i32[2] )
            {
              v6 = v2[1].m128i_i64[0];
              if ( v6 < src[1].m128i_i64[0] )
                goto LABEL_17;
              if ( v6 == src[1].m128i_i64[0] )
              {
                v9 = v2[1].m128i_u64[1];
                if ( v9 < src[1].m128i_i64[1] )
                  goto LABEL_18;
              }
            }
          }
        }
        v7 = v2 + 2;
        result = sub_19E2710(v2);
LABEL_12:
        v2 = v7;
        if ( a2 == v7 )
          return result;
      }
      v4 = v2->m128i_i32[1];
LABEL_15:
      v5 = v2->m128i_i32[2];
LABEL_16:
      v6 = v2[1].m128i_i64[0];
LABEL_17:
      v9 = v2[1].m128i_u64[1];
LABEL_18:
      v7 = v2 + 2;
      if ( src != v2 )
      {
        v11 = v5;
        v10 = v6;
        v12 = v4;
        result = (__int64)memmove(&src[2], src, (char *)v2 - (char *)src);
        v5 = v11;
        v6 = v10;
        v4 = v12;
      }
      src->m128i_i32[0] = v3;
      src->m128i_i32[1] = v4;
      src->m128i_i32[2] = v5;
      src[1].m128i_i64[0] = v6;
      src[1].m128i_i64[1] = v9;
      goto LABEL_12;
    }
  }
  return result;
}
