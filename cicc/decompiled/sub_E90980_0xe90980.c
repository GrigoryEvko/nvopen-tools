// Function: sub_E90980
// Address: 0xe90980
//
unsigned __int64 __fastcall sub_E90980(unsigned __int64 *a1, char *a2, const __m128i *a3, unsigned __int64 *a4)
{
  char *v6; // rsi
  char *v7; // r14
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  bool v10; // cf
  unsigned __int64 result; // rax
  char *v12; // r9
  __int64 v13; // r8
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r13
  __m128i *v16; // r9
  __m128i v17; // xmm0
  unsigned __int64 v18; // rdx
  char *v19; // rax
  char *v20; // rax
  __int64 v21; // rdx
  int v22; // ecx
  __int64 v23; // r15
  unsigned __int64 *v24; // [rsp+0h] [rbp-50h]
  const __m128i *v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v6 = (char *)a1[1];
  v7 = (char *)*a1;
  v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v6[-*a1] >> 3);
  if ( v8 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v7) >> 3);
  v10 = __CFADD__(v9, v8);
  result = v9 - 0x5555555555555555LL * ((v6 - v7) >> 3);
  v12 = (char *)(a2 - v7);
  if ( v10 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !result )
    {
      v13 = 24;
      v14 = 0;
      v15 = 0;
      goto LABEL_7;
    }
    if ( result > 0x555555555555555LL )
      result = 0x555555555555555LL;
    v23 = 24 * result;
  }
  v24 = a4;
  v25 = a3;
  result = sub_22077B0(v23);
  v12 = (char *)(a2 - v7);
  a3 = v25;
  a4 = v24;
  v15 = result;
  v14 = result + v23;
  v13 = result + 24;
LABEL_7:
  v16 = (__m128i *)&v12[v15];
  if ( v16 )
  {
    result = *a4;
    v17 = _mm_loadu_si128(a3);
    v16[1].m128i_i64[0] = *a4;
    *v16 = v17;
  }
  if ( a2 != v7 )
  {
    v18 = v15;
    v19 = v7;
    do
    {
      if ( v18 )
      {
        *(_DWORD *)v18 = *(_DWORD *)v19;
        *(_QWORD *)(v18 + 8) = *((_QWORD *)v19 + 1);
        *(_QWORD *)(v18 + 16) = *((_QWORD *)v19 + 2);
      }
      v19 += 24;
      v18 += 24LL;
    }
    while ( v19 != a2 );
    result = (unsigned __int64)(a2 - 24 - v7) >> 3;
    v13 = v15 + 8 * result + 48;
  }
  if ( a2 != v6 )
  {
    v20 = a2;
    v21 = v13;
    do
    {
      v22 = *(_DWORD *)v20;
      v20 += 24;
      v21 += 24;
      *(_DWORD *)(v21 - 24) = v22;
      *(_QWORD *)(v21 - 16) = *((_QWORD *)v20 - 2);
      *(_QWORD *)(v21 - 8) = *((_QWORD *)v20 - 1);
    }
    while ( v20 != v6 );
    result = (unsigned __int64)(v20 - a2 - 24) >> 3;
    v13 += 8 * result + 24;
  }
  if ( v7 )
  {
    v26 = v13;
    result = j_j___libc_free_0(v7, a1[2] - (_QWORD)v7);
    v13 = v26;
  }
  *a1 = v15;
  a1[2] = v14;
  a1[1] = v13;
  return result;
}
