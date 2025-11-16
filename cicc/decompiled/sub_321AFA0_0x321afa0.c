// Function: sub_321AFA0
// Address: 0x321afa0
//
void __fastcall sub_321AFA0(__m128i *a1, const __m128i *a2)
{
  const __m128i *v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  bool v5; // al
  const __m128i *v6; // r12
  __int64 v7; // rcx
  __int64 *v8; // r15
  bool v9; // dl
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
          sub_AF47B0((__int64)v12, *(unsigned __int64 **)(v3 + 16), *(unsigned __int64 **)(v3 + 24));
          sub_AF47B0((__int64)v15, *(unsigned __int64 **)(v4 + 16), *(unsigned __int64 **)(v4 + 24));
          v5 = v17;
          v3 = v2->m128i_i64[1];
          if ( v14 )
          {
            if ( !v17 )
            {
              v10 = v2->m128i_i64[0];
              goto LABEL_11;
            }
            v5 = v13 < v16;
          }
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
LABEL_11:
          v6 = v2;
          v7 = v2[-1].m128i_i64[1];
          v8 = (__int64 *)v2;
          if ( !v7 )
            goto LABEL_16;
          while ( 1 )
          {
            if ( !v3 )
              goto LABEL_16;
            v11 = v7;
            sub_AF47B0((__int64)v12, *(unsigned __int64 **)(v3 + 16), *(unsigned __int64 **)(v3 + 24));
            sub_AF47B0((__int64)v15, *(unsigned __int64 **)(v11 + 16), *(unsigned __int64 **)(v11 + 24));
            v9 = v17;
            if ( v14 )
              break;
            while ( 1 )
            {
LABEL_14:
              --v6;
              if ( !v9 )
                goto LABEL_5;
              v7 = v6[-1].m128i_i64[1];
              v8 = (__int64 *)v6;
              v6[1] = _mm_loadu_si128(v6);
              if ( v7 )
                break;
LABEL_16:
              v9 = v7 != 0;
            }
          }
          if ( v17 )
          {
            v9 = v13 < v16;
            goto LABEL_14;
          }
LABEL_5:
          v8[1] = v3;
          ++v2;
          *v8 = v10;
          if ( a2 == v2 )
            return;
        }
      }
    }
  }
}
