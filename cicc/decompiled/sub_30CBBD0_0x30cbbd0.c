// Function: sub_30CBBD0
// Address: 0x30cbbd0
//
__m128i *__fastcall sub_30CBBD0(__m128i *a1, __int64 a2)
{
  size_t v3; // rax
  char *v4; // r15
  char *v5; // r15
  size_t v6; // rdx
  __m128i *v7; // rax
  size_t v8; // rcx
  __m128i *v9; // r9
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rdx
  __int64 v16[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v18; // [rsp+30h] [rbp-70h] BYREF
  size_t v19; // [rsp+38h] [rbp-68h]
  __m128i v20; // [rsp+40h] [rbp-60h] BYREF
  char *v21; // [rsp+50h] [rbp-50h] BYREF
  size_t v22; // [rsp+58h] [rbp-48h]
  _QWORD v23[8]; // [rsp+60h] [rbp-40h] BYREF

  switch ( HIDWORD(a2) )
  {
    case 0:
      v3 = 13;
      v4 = "always-inline";
      v21 = (char *)v23;
      break;
    case 1:
      v4 = "cgscc-inline";
      goto LABEL_30;
    case 2:
      v4 = "early-inline";
      goto LABEL_30;
    case 3:
      v4 = "module-inline";
      goto LABEL_30;
    case 4:
      v4 = "ml-inline";
      goto LABEL_30;
    case 5:
      v4 = "replay-cgscc-inline";
      goto LABEL_30;
    case 6:
      v4 = "replay-sample-profile-inline";
      goto LABEL_30;
    case 7:
      v4 = "sample-profile-inline";
LABEL_30:
      v21 = (char *)v23;
      v3 = strlen(v4);
      break;
    default:
LABEL_39:
      BUG();
  }
  sub_30CA380((__int64 *)&v21, v4, (__int64)&v4[v3]);
  switch ( (int)a2 )
  {
    case 0:
      v6 = 4;
      v5 = "main";
      v16[0] = (__int64)v17;
      goto LABEL_6;
    case 1:
    case 3:
      v5 = "prelink";
      goto LABEL_5;
    case 2:
    case 4:
      v5 = "postlink";
LABEL_5:
      v16[0] = (__int64)v17;
      v6 = strlen(v5);
LABEL_6:
      sub_30CA380(v16, v5, (__int64)&v5[v6]);
      if ( v16[1] == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v7 = (__m128i *)sub_2241490((unsigned __int64 *)v16, "-", 1u);
      v18 = &v20;
      if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
      {
        v20 = _mm_loadu_si128(v7 + 1);
      }
      else
      {
        v18 = (__m128i *)v7->m128i_i64[0];
        v20.m128i_i64[0] = v7[1].m128i_i64[0];
      }
      v8 = v7->m128i_u64[1];
      v7[1].m128i_i8[0] = 0;
      v19 = v8;
      v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
      v9 = v18;
      v7->m128i_i64[1] = 0;
      v10 = 15;
      v11 = 15;
      if ( v9 != &v20 )
        v11 = v20.m128i_i64[0];
      if ( v19 + v22 <= v11 )
        goto LABEL_15;
      if ( v21 != (char *)v23 )
        v10 = v23[0];
      if ( v19 + v22 <= v10 )
      {
        v12 = (__m128i *)sub_2241130((unsigned __int64 *)&v21, 0, 0, v9, v19);
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        v13 = v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
          goto LABEL_16;
      }
      else
      {
LABEL_15:
        v12 = (__m128i *)sub_2241490((unsigned __int64 *)&v18, v21, v22);
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        v13 = v12->m128i_i64[0];
        v14 = v12 + 1;
        if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        {
LABEL_16:
          a1->m128i_i64[0] = v13;
          a1[1].m128i_i64[0] = v12[1].m128i_i64[0];
          goto LABEL_17;
        }
      }
      a1[1] = _mm_loadu_si128(v12 + 1);
LABEL_17:
      a1->m128i_i64[1] = v12->m128i_i64[1];
      v12->m128i_i64[0] = (__int64)v14;
      v12->m128i_i64[1] = 0;
      v12[1].m128i_i8[0] = 0;
      if ( v18 != &v20 )
        j_j___libc_free_0((unsigned __int64)v18);
      if ( (_QWORD *)v16[0] != v17 )
        j_j___libc_free_0(v16[0]);
      if ( v21 != (char *)v23 )
        j_j___libc_free_0((unsigned __int64)v21);
      return a1;
    default:
      goto LABEL_39;
  }
}
