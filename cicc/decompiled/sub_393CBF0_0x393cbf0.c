// Function: sub_393CBF0
// Address: 0x393cbf0
//
__int64 *__fastcall sub_393CBF0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rdx
  const __m128i *v3; // rcx
  const __m128i *v4; // r8
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r15
  __m128i *v9; // rdx
  const __m128i *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  __int64 v18; // [rsp+28h] [rbp-38h]

  sub_393C750(a2);
  v3 = (const __m128i *)a2[10];
  v4 = (const __m128i *)a2[9];
  v5 = (char *)v3 - (char *)v4;
  if ( v3 != v4 )
  {
    if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a2, a2, v2);
    v6 = sub_22077B0(v5);
    v3 = (const __m128i *)a2[10];
    v4 = (const __m128i *)a2[9];
    v7 = v6;
    v8 = v6 + v5;
    if ( v3 != v4 )
      goto LABEL_4;
LABEL_15:
    v14 = v7;
    goto LABEL_9;
  }
  v7 = 0;
  v8 = 0;
  if ( v3 == v4 )
    goto LABEL_15;
LABEL_4:
  v9 = (__m128i *)v7;
  v10 = v4;
  do
  {
    if ( v9 )
    {
      *v9 = _mm_loadu_si128(v10);
      v9[1].m128i_i64[0] = v10[1].m128i_i64[0];
    }
    v10 = (const __m128i *)((char *)v10 + 24);
    v9 = (__m128i *)((char *)v9 + 24);
  }
  while ( v10 != v3 );
  v14 = v7 + 8 * ((unsigned __int64)((char *)&v10[-2].m128i_u64[1] - (char *)v4) >> 3) + 24;
LABEL_9:
  v15 = a2[12];
  v11 = a2[15];
  v16 = a2[13];
  v17 = a2[16];
  v18 = a2[14];
  v12 = sub_22077B0(0x48u);
  if ( v12 )
  {
    *(_DWORD *)v12 = 0;
    *(_QWORD *)(v12 + 8) = v7;
    *(_QWORD *)(v12 + 16) = v14;
    *(_QWORD *)(v12 + 24) = v8;
    *(_QWORD *)(v12 + 32) = v15;
    *(_QWORD *)(v12 + 40) = v16;
    *(_QWORD *)(v12 + 48) = v17;
    *(_QWORD *)(v12 + 56) = v18;
    *(_QWORD *)(v12 + 64) = v11;
    *a1 = v12;
  }
  else
  {
    *a1 = 0;
    if ( v7 )
      j_j___libc_free_0(v7);
  }
  return a1;
}
