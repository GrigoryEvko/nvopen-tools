// Function: sub_8332F0
// Address: 0x8332f0
//
__int64 __fastcall sub_8332F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdi
  const __m128i *v14; // r12
  __m128i *v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 *v19; // r9
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 v25; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 1016);
  v8 = *(_QWORD *)(a1 + 1024);
  v9 = *(_QWORD *)(a1 + 1008);
  v10 = *(_DWORD *)(a1 + 4);
  v21 = v7;
  v24 = v9;
  if ( v7 <= 1 )
  {
    v11 = a1 + 8;
    v23 = a1 + 8;
    if ( v10 && v11 != v9 )
    {
      v22 = 2;
      v13 = 80;
      goto LABEL_5;
    }
    v22 = 2;
  }
  else
  {
    v11 = v7 + (v7 >> 1) + 1;
    v22 = v11;
    if ( v10 && v9 != a1 + 8 )
    {
      v12 = v7 + (v7 >> 1) + 1;
      goto LABEL_4;
    }
    v12 = v7 + (v7 >> 1) + 1;
    if ( v11 > 25 )
    {
LABEL_4:
      v13 = 40 * v12;
LABEL_5:
      v23 = sub_823970(v13);
      goto LABEL_6;
    }
    v23 = a1 + 8;
  }
  *(_DWORD *)(a1 + 4) = 1;
LABEL_6:
  v14 = (const __m128i *)v24;
  if ( v23 != v24 )
  {
    v15 = (__m128i *)v23;
    v16 = 0;
    if ( v8 > 0 )
    {
      do
      {
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v14);
          v15[1] = _mm_loadu_si128(v14 + 1);
          v15[2].m128i_i64[0] = v14[2].m128i_i64[0];
        }
        a5 = v14[2].m128i_i64[0];
        if ( a5 )
        {
          v25 = v14[2].m128i_i64[0];
          sub_823A00(*(_QWORD *)a5, 16LL * (unsigned int)(*(_DWORD *)(a5 + 8) + 1), v11, v7, a5, a6);
          sub_823A00(v25, 16, v17, v18, v25, v19);
        }
        ++v16;
        v15 = (__m128i *)((char *)v15 + 40);
        v14 = (const __m128i *)((char *)v14 + 40);
      }
      while ( v16 != v8 );
    }
    if ( v24 == a1 + 8 )
      *(_DWORD *)(a1 + 4) = 0;
    else
      sub_823A00(v24, 40 * v21, v11, v7, a5, a6);
  }
  *(_QWORD *)(a1 + 1008) = v23;
  *(_QWORD *)(a1 + 1016) = v22;
  return v22;
}
