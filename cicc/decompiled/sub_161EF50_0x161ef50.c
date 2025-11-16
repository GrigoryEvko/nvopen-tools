// Function: sub_161EF50
// Address: 0x161ef50
//
void __fastcall sub_161EF50(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int32 v5; // eax
  __int64 v7; // rdx
  __m128i *v8; // rax
  __int64 v9; // rdx
  __m128i *i; // rdx
  unsigned int v11; // eax
  unsigned int v12; // ebx
  __int8 v13; // al
  __int64 v14; // rdi
  __int64 v15; // rax
  bool v16; // zf
  __m128i *v17; // rax
  __int64 v18; // rdx
  __m128i *j; // rdx
  __m128i *v20; // r12
  __int64 v21; // rax
  __m128i *v22; // rax

  v5 = (unsigned __int32)a1[1].m128i_i32[2] >> 1;
  if ( !v5 )
    return;
  if ( (_BYTE)a2 )
  {
    sub_161EAA0(a1, a2, a3, a4, a5);
    return;
  }
  ++a1[1].m128i_i64[0];
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    v8 = (__m128i *)&a1[2];
    v9 = 96;
    goto LABEL_9;
  }
  v7 = a1[2].m128i_u32[2];
  if ( 4 * v5 >= (unsigned int)v7 || (unsigned int)v7 <= 0x40 )
  {
    v8 = (__m128i *)a1[2].m128i_i64[0];
    v9 = 24 * v7;
LABEL_9:
    for ( i = (__m128i *)((char *)v8 + v9); i != v8; v8 = (__m128i *)((char *)v8 + 24) )
      v8->m128i_i64[0] = -4;
    a1[1].m128i_i64[1] &= 1uLL;
    return;
  }
  v11 = v5 - 1;
  if ( !v11 )
  {
    j___libc_free_0(a1[2].m128i_i64[0]);
    a1[1].m128i_i8[8] |= 1u;
LABEL_20:
    v16 = (a1[1].m128i_i64[1] & 1) == 0;
    a1[1].m128i_i64[1] &= 1uLL;
    if ( v16 )
    {
      v17 = (__m128i *)a1[2].m128i_i64[0];
      v18 = 24LL * a1[2].m128i_u32[2];
    }
    else
    {
      v17 = (__m128i *)&a1[2];
      v18 = 96;
    }
    for ( j = (__m128i *)((char *)v17 + v18); j != v17; v17 = (__m128i *)((char *)v17 + 24) )
    {
      if ( v17 )
        v17->m128i_i64[0] = -4;
    }
    return;
  }
  _BitScanReverse(&v11, v11);
  v12 = 1 << (33 - (v11 ^ 0x1F));
  if ( v12 - 5 <= 0x3A )
  {
    v12 = 64;
    j___libc_free_0(a1[2].m128i_i64[0]);
    v13 = a1[1].m128i_i8[8];
    v14 = 1536;
LABEL_19:
    a1[1].m128i_i8[8] = v13 & 0xFE;
    v15 = sub_22077B0(v14);
    a1[2].m128i_i32[2] = v12;
    a1[2].m128i_i64[0] = v15;
    goto LABEL_20;
  }
  if ( (_DWORD)v7 != v12 )
  {
    j___libc_free_0(a1[2].m128i_i64[0]);
    v13 = a1[1].m128i_i8[8] | 1;
    a1[1].m128i_i8[8] = v13;
    if ( v12 <= 4 )
      goto LABEL_20;
    v14 = 24LL * v12;
    goto LABEL_19;
  }
  v16 = (a1[1].m128i_i64[1] & 1) == 0;
  a1[1].m128i_i64[1] &= 1uLL;
  if ( v16 )
  {
    v20 = (__m128i *)a1[2].m128i_i64[0];
    v21 = 24 * v7;
  }
  else
  {
    v20 = (__m128i *)&a1[2];
    v21 = 96;
  }
  v22 = (__m128i *)((char *)v20 + v21);
  do
  {
    if ( v20 )
      v20->m128i_i64[0] = -4;
    v20 = (__m128i *)((char *)v20 + 24);
  }
  while ( v22 != v20 );
}
