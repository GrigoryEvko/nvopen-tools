// Function: sub_357CB90
// Address: 0x357cb90
//
void __fastcall sub_357CB90(__m128i *a1)
{
  __m128i *v1; // r15
  __m128i *v2; // r13
  size_t v3; // r14
  __m128i *v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rax
  size_t v7; // r9
  unsigned __int64 *v8; // rcx
  __m128i **v9; // rbx
  size_t v10; // rdx
  signed __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  size_t v16; // [rsp+10h] [rbp-70h]
  unsigned __int64 *v17; // [rsp+18h] [rbp-68h]
  __m128i *v18; // [rsp+20h] [rbp-60h]
  size_t v19; // [rsp+28h] [rbp-58h]
  __m128i v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+40h] [rbp-40h]

  v1 = a1 + 1;
  v2 = (__m128i *)a1->m128i_i64[0];
  v18 = &v20;
  if ( (__m128i *)a1->m128i_i64[0] == &a1[1] )
  {
    v2 = &v20;
    v20 = _mm_loadu_si128(a1 + 1);
  }
  else
  {
    v18 = (__m128i *)a1->m128i_i64[0];
    v20.m128i_i64[0] = a1[1].m128i_i64[0];
  }
  v3 = a1->m128i_u64[1];
  a1->m128i_i64[0] = (__int64)v1;
  v4 = a1 + 1;
  a1->m128i_i64[1] = 0;
  a1[1].m128i_i8[0] = 0;
  v19 = v3;
  v21 = a1[2].m128i_i64[0];
  while ( 1 )
  {
    v7 = v4[-3].m128i_u64[0];
    v8 = (unsigned __int64 *)v4[-4].m128i_i64[1];
    v9 = (__m128i **)&v4[-1];
    v10 = v7;
    if ( v3 <= v7 )
      v10 = v3;
    if ( v10 )
    {
      v16 = v4[-3].m128i_u64[0];
      v17 = (unsigned __int64 *)v4[-4].m128i_i64[1];
      LODWORD(v11) = memcmp(v2, v8, v10);
      v8 = v17;
      v7 = v16;
      if ( (_DWORD)v11 )
        goto LABEL_14;
    }
    v11 = v3 - v7;
    if ( (__int64)(v3 - v7) >= 0x80000000LL )
      break;
    if ( v11 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_14:
      if ( (int)v11 >= 0 )
        break;
    }
    v12 = &v4[-3].m128i_i64[1];
    if ( v8 == &v4[-3].m128i_u64[1] )
    {
      if ( v7 )
      {
        if ( v7 == 1 )
          v1->m128i_i8[0] = *(_BYTE *)v12;
        else
          memcpy(v1, &v4[-3].m128i_u64[1], v7);
        v7 = v4[-3].m128i_u64[0];
        v1 = (__m128i *)v12[3];
      }
      v12[4] = v7;
      v1->m128i_i8[v7] = 0;
      v1 = (__m128i *)v4[-4].m128i_i64[1];
    }
    else
    {
      if ( v4 == v1 )
      {
        v13 = v4[-3].m128i_i64[0];
        v12[3] = (__int64)v8;
        v12[4] = v13;
        v4->m128i_i64[0] = v4[-3].m128i_i64[1];
      }
      else
      {
        v5 = v4[-3].m128i_i64[0];
        v6 = v4->m128i_i64[0];
        v12[3] = (__int64)v8;
        v12[4] = v5;
        v4->m128i_i64[0] = v4[-3].m128i_i64[1];
        if ( v1 )
        {
          v4[-4].m128i_i64[1] = (__int64)v1;
          *v12 = v6;
          goto LABEL_7;
        }
      }
      v4[-4].m128i_i64[1] = (__int64)&v4[-3].m128i_i64[1];
      v1 = (__m128i *)((char *)v4 - 40);
    }
LABEL_7:
    v4[-3].m128i_i64[0] = 0;
    v4 = (__m128i *)((char *)v4 - 40);
    v1->m128i_i8[0] = 0;
    v2 = v18;
    v3 = v19;
    v12[7] = v12[2];
    v1 = (__m128i *)*(v12 - 2);
  }
  if ( v2 == &v20 )
  {
    if ( v3 )
    {
      if ( v3 == 1 )
        v1->m128i_i8[0] = v20.m128i_i8[0];
      else
        memcpy(v1, &v20, v3);
      v3 = v19;
      v1 = *v9;
    }
    v9[1] = (__m128i *)v3;
    v1->m128i_i8[v3] = 0;
    v1 = v18;
  }
  else
  {
    v14 = v20.m128i_i64[0];
    if ( v4 == v1 )
    {
      *v9 = v2;
      v9[1] = (__m128i *)v3;
      v4->m128i_i64[0] = v14;
    }
    else
    {
      v15 = v4->m128i_i64[0];
      *v9 = v2;
      v9[1] = (__m128i *)v3;
      v4->m128i_i64[0] = v14;
      if ( v1 )
      {
        v18 = v1;
        v20.m128i_i64[0] = v15;
        goto LABEL_27;
      }
    }
    v18 = &v20;
    v1 = &v20;
  }
LABEL_27:
  v1->m128i_i8[0] = 0;
  v9[4] = (__m128i *)v21;
  if ( v18 != &v20 )
    j_j___libc_free_0((unsigned __int64)v18);
}
