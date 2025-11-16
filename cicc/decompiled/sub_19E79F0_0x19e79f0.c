// Function: sub_19E79F0
// Address: 0x19e79f0
//
unsigned __int64 __fastcall sub_19E79F0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int8 *v5; // rax
  __int64 v6; // rbx
  __int8 *v7; // rdi
  __int64 v8; // rax
  __m128i *v9; // r15
  char *v10; // r14
  char *v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // r15
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  char *v18; // r8
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v22; // [rsp+10h] [rbp-110h] BYREF
  __m128i v23; // [rsp+20h] [rbp-100h] BYREF
  __m128i v24; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v25; // [rsp+40h] [rbp-E0h]
  __int64 v26; // [rsp+58h] [rbp-C8h] BYREF
  __int64 src; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-B8h] BYREF
  __m128i dest[4]; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v30; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v31; // [rsp+C0h] [rbp-60h]
  __m128i v32; // [rsp+D0h] [rbp-50h]
  __int64 v33; // [rsp+E0h] [rbp-40h]
  unsigned __int64 v34; // [rsp+E8h] [rbp-38h]

  v3 = sub_15AF870();
  v4 = *a1;
  v34 = v3;
  v26 = 0;
  v5 = sub_19E7740(dest, &v26, dest[0].m128i_i8, (unsigned __int64)&v30, v4);
  v6 = v26;
  v7 = v5;
  v8 = *a2;
  v9 = (__m128i *)(v7 + 8);
  src = *a2;
  if ( v7 + 8 <= (__int8 *)&v30 )
  {
    *(_QWORD *)v7 = v8;
  }
  else
  {
    v10 = (char *)((char *)&v30 - v7);
    memcpy(v7, &src, (char *)&v30 - v7);
    if ( v6 )
    {
      v6 += 64;
      sub_1593A20((unsigned __int64 *)&v30, dest);
    }
    else
    {
      v6 = 64;
      sub_15938B0((unsigned __int64 *)&v22, dest[0].m128i_i64, v34);
      v19 = _mm_loadu_si128(&v23);
      v20 = _mm_loadu_si128(&v24);
      v30 = _mm_loadu_si128(&v22);
      v33 = v25;
      v31 = v19;
      v32 = v20;
    }
    v9 = (__m128i *)((char *)dest + 8LL - (_QWORD)v10);
    if ( v9 > &v30 )
LABEL_5:
      abort();
    memcpy(dest, (char *)&src + (_QWORD)v10, 8LL - (_QWORD)v10);
  }
  v11 = &v9->m128i_i8[8];
  v12 = *a3;
  v28 = *a3;
  if ( &v9->m128i_u64[1] > (unsigned __int64 *)&v30 )
  {
    memcpy(v9, &v28, (char *)&v30 - (char *)v9);
    if ( v6 )
    {
      v6 += 64;
      sub_1593A20((unsigned __int64 *)&v30, dest);
      v18 = (char *)((char *)&v30 - (char *)v9);
    }
    else
    {
      v6 = 64;
      sub_15938B0((unsigned __int64 *)&v22, dest[0].m128i_i64, v34);
      v15 = _mm_loadu_si128(&v22);
      v16 = _mm_loadu_si128(&v23);
      v17 = _mm_loadu_si128(&v24);
      v33 = v25;
      v18 = (char *)((char *)&v30 - (char *)v9);
      v30 = v15;
      v31 = v16;
      v32 = v17;
    }
    v11 = &dest[0].m128i_i8[8LL - (_QWORD)v18];
    if ( v11 > (char *)&v30 )
      goto LABEL_5;
    v13 = 8LL - (_QWORD)v18;
    memcpy(dest, (char *)&v28 + (_QWORD)v18, 8LL - (_QWORD)v18);
    if ( !v6 )
      return sub_1593600(dest, v13, v34);
  }
  else
  {
    v9->m128i_i64[0] = v12;
    v13 = v11 - (char *)dest;
    if ( !v6 )
      return sub_1593600(&dest[0], v13, v34);
  }
  sub_19E16D0(dest[0].m128i_i8, v11, v30.m128i_i8);
  sub_1593A20((unsigned __int64 *)&v30, dest);
  return sub_19E45B0(&v30, v13 + v6);
}
