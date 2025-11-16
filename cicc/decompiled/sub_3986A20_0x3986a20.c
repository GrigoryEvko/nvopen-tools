// Function: sub_3986A20
// Address: 0x3986a20
//
void __fastcall sub_3986A20(__m128i *a1, const __m128i *a2)
{
  const __m128i *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  bool v5; // al
  const __m128i *i; // r12
  bool v7; // dl
  __int64 v8; // rcx
  __int64 *v9; // r15
  __int64 v10; // [rsp+10h] [rbp-90h]
  __int64 v11; // [rsp+28h] [rbp-78h]
  char v12[8]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v13; // [rsp+38h] [rbp-68h]
  char v14; // [rsp+40h] [rbp-60h]
  char v15[8]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v16; // [rsp+58h] [rbp-48h]
  bool v17; // [rsp+60h] [rbp-40h]

  if ( a1 != a2 )
  {
    v2 = a1 + 1;
    if ( a2 != &a1[1] )
    {
      while ( 1 )
      {
        v3 = v2->m128i_i64[1];
        v4 = a1->m128i_i64[1];
        if ( v4 && v3 )
        {
          sub_15B1350((__int64)v12, *(unsigned __int64 **)(v3 + 24), *(unsigned __int64 **)(v3 + 32));
          sub_15B1350((__int64)v15, *(unsigned __int64 **)(v4 + 24), *(unsigned __int64 **)(v4 + 32));
          if ( v14 )
          {
            if ( !v17 )
            {
              v3 = v2->m128i_i64[1];
              v10 = v2->m128i_i64[0];
              goto LABEL_13;
            }
            v5 = v13 < v16;
          }
          else
          {
            v5 = v17;
          }
          v3 = v2->m128i_i64[1];
        }
        else
        {
          v5 = v4 != 0;
        }
        v10 = v2->m128i_i64[0];
        if ( v5 )
        {
          if ( a1 != v2 )
            memmove(&a1[1], a1, (char *)v2 - (char *)a1);
          ++v2;
          a1->m128i_i64[0] = v10;
          a1->m128i_i64[1] = v3;
          if ( a2 == v2 )
            return;
        }
        else
        {
LABEL_13:
          for ( i = v2; ; i[1] = _mm_loadu_si128(i) )
          {
            v8 = i[-1].m128i_i64[1];
            v9 = (__int64 *)i;
            if ( !v8 || !v3 )
            {
              v7 = v8 != 0;
              goto LABEL_15;
            }
            v11 = i[-1].m128i_i64[1];
            sub_15B1350((__int64)v12, *(unsigned __int64 **)(v3 + 24), *(unsigned __int64 **)(v3 + 32));
            sub_15B1350((__int64)v15, *(unsigned __int64 **)(v11 + 24), *(unsigned __int64 **)(v11 + 32));
            if ( !v14 )
            {
              v7 = v17;
              goto LABEL_15;
            }
            if ( !v17 )
              break;
            v7 = v13 < v16;
LABEL_15:
            --i;
            if ( !v7 )
              break;
          }
          v9[1] = v3;
          ++v2;
          *v9 = v10;
          if ( a2 == v2 )
            return;
        }
      }
    }
  }
}
