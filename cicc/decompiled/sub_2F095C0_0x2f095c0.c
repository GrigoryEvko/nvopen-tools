// Function: sub_2F095C0
// Address: 0x2f095c0
//
void __fastcall sub_2F095C0(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rbx
  unsigned __int64 v6; // r14
  signed __int64 v7; // rcx
  unsigned __int64 v8; // r15
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rcx
  unsigned __int64 *v14; // rsi
  unsigned __int64 *v15; // rdi
  unsigned __int64 v16; // rdi
  signed __int64 v17; // r14
  __int64 v18; // r14
  __int64 i; // r15
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r14
  unsigned __int64 *v24; // rbx
  unsigned __int64 v25; // rax
  unsigned __int64 *v26; // rsi
  unsigned __int64 *v27; // rdi
  unsigned __int64 v28; // rbx
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  signed __int64 v32; // [rsp+18h] [rbp-38h]

  if ( a2 != a1 )
  {
    v3 = a2[1];
    v4 = *a2;
    v6 = *a1;
    v7 = v3 - *a2;
    v32 = v7;
    if ( a1[2] - *a1 < v7 )
    {
      if ( v7 )
      {
        if ( (unsigned __int64)v7 > 0x7FFFFFFFFFFFFFC0LL )
          sub_4261EA(a1, a2, a3);
        v18 = sub_22077B0(v7);
      }
      else
      {
        v18 = 0;
      }
      for ( i = v18; v3 != v4; i += 64 )
      {
        if ( i )
        {
          *(_QWORD *)i = *(_QWORD *)v4;
          *(_QWORD *)(i + 8) = i + 24;
          sub_2F07250((__int64 *)(i + 8), *(_BYTE **)(v4 + 8), *(_QWORD *)(v4 + 8) + *(_QWORD *)(v4 + 16));
          *(__m128i *)(i + 40) = _mm_loadu_si128((const __m128i *)(v4 + 40));
          *(_DWORD *)(i + 56) = *(_DWORD *)(v4 + 56);
        }
        v4 += 64LL;
      }
      v20 = a1[1];
      v21 = *a1;
      if ( v20 != *a1 )
      {
        do
        {
          v22 = *(_QWORD *)(v21 + 8);
          if ( v22 != v21 + 24 )
            j_j___libc_free_0(v22);
          v21 += 64LL;
        }
        while ( v20 != v21 );
        v21 = *a1;
      }
      if ( v21 )
        j_j___libc_free_0(v21);
      *a1 = v18;
      v17 = v32 + v18;
      a1[2] = v17;
      goto LABEL_12;
    }
    v8 = a1[1];
    v9 = v8 - v6;
    v10 = v8 - v6;
    if ( v7 > v8 - v6 )
    {
      v31 = v9 >> 6;
      if ( v9 > 0 )
      {
        v23 = (unsigned __int64 *)(v6 + 8);
        v24 = (unsigned __int64 *)(v4 + 8);
        do
        {
          v25 = *(v24 - 1);
          v26 = v24;
          v27 = v23;
          v24 += 8;
          v23 += 8;
          *(v23 - 9) = v25;
          sub_2240AE0(v27, v26);
          *((__m128i *)v23 - 2) = _mm_loadu_si128((const __m128i *)v24 - 2);
          *((_DWORD *)v23 - 4) = *((_DWORD *)v24 - 4);
          --v31;
        }
        while ( v31 );
        v8 = a1[1];
        v6 = *a1;
        v3 = a2[1];
        v4 = *a2;
        v10 = v8 - *a1;
      }
      v28 = v10 + v4;
      v17 = v32 + v6;
      if ( v28 == v3 )
        goto LABEL_12;
      do
      {
        if ( v8 )
        {
          *(_QWORD *)v8 = *(_QWORD *)v28;
          *(_QWORD *)(v8 + 8) = v8 + 24;
          sub_2F07250((__int64 *)(v8 + 8), *(_BYTE **)(v28 + 8), *(_QWORD *)(v28 + 8) + *(_QWORD *)(v28 + 16));
          *(__m128i *)(v8 + 40) = _mm_loadu_si128((const __m128i *)(v28 + 40));
          *(_DWORD *)(v8 + 56) = *(_DWORD *)(v28 + 56);
        }
        v28 += 64LL;
        v8 += 64LL;
      }
      while ( v28 != v3 );
    }
    else
    {
      v11 = (unsigned __int64 *)(v6 + 8);
      v12 = (unsigned __int64 *)(v4 + 8);
      v30 = v7 >> 6;
      if ( v7 <= 0 )
        goto LABEL_10;
      do
      {
        v13 = *(v12 - 1);
        v14 = v12;
        v15 = v11;
        v12 += 8;
        v11 += 8;
        *(v11 - 9) = v13;
        sub_2240AE0(v15, v14);
        *((__m128i *)v11 - 2) = _mm_loadu_si128((const __m128i *)v12 - 2);
        *((_DWORD *)v11 - 4) = *((_DWORD *)v12 - 4);
        --v30;
      }
      while ( v30 );
      v6 += v32;
      while ( v8 != v6 )
      {
        v16 = *(_QWORD *)(v6 + 8);
        if ( v16 != v6 + 24 )
          j_j___libc_free_0(v16);
        v6 += 64LL;
LABEL_10:
        ;
      }
    }
    v17 = *a1 + v32;
LABEL_12:
    a1[1] = v17;
  }
}
