// Function: sub_EF8D70
// Address: 0xef8d70
//
__int64 __fastcall sub_EF8D70(int *a1, __int64 a2)
{
  __int64 v3; // rax
  const __m128i *v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rax
  __m128i *v7; // rax
  __m128i *v8; // rcx
  __m128i *v9; // rdx
  __m128i *v10; // rax
  __m128i *v11; // rdx
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdi
  int *v15; // r12
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rax
  const __m128i *v19; // rdi
  __m128i *v20; // rax
  __m128i *v21; // rcx
  __m128i *v22; // rdx
  __m128i *v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rdi

  v3 = sub_22077B0(88);
  v4 = (const __m128i *)*((_QWORD *)a1 + 7);
  v5 = v3;
  v6 = *((_QWORD *)a1 + 4);
  *(_DWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 32) = v6;
  *(_QWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 64) = v5 + 48;
  *(_QWORD *)(v5 + 72) = v5 + 48;
  *(_QWORD *)(v5 + 80) = 0;
  if ( v4 )
  {
    v7 = sub_EF89A0(v4, v5 + 48);
    v8 = v7;
    do
    {
      v9 = v7;
      v7 = (__m128i *)v7[1].m128i_i64[0];
    }
    while ( v7 );
    *(_QWORD *)(v5 + 64) = v9;
    v10 = v8;
    do
    {
      v11 = v10;
      v10 = (__m128i *)v10[1].m128i_i64[1];
    }
    while ( v10 );
    v12 = *((_QWORD *)a1 + 10);
    *(_QWORD *)(v5 + 72) = v11;
    *(_QWORD *)(v5 + 56) = v8;
    *(_QWORD *)(v5 + 80) = v12;
  }
  v13 = *a1;
  v14 = *((_QWORD *)a1 + 3);
  *(_QWORD *)(v5 + 8) = a2;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)v5 = v13;
  *(_QWORD *)(v5 + 24) = 0;
  if ( v14 )
    *(_QWORD *)(v5 + 24) = sub_EF8D70(v14, v5);
  v15 = (int *)*((_QWORD *)a1 + 2);
  if ( v15 )
  {
    v16 = v5;
    do
    {
      v17 = v16;
      v16 = sub_22077B0(88);
      v18 = *((_QWORD *)v15 + 4);
      *(_DWORD *)(v16 + 48) = 0;
      *(_QWORD *)(v16 + 32) = v18;
      *(_QWORD *)(v16 + 56) = 0;
      *(_QWORD *)(v16 + 64) = v16 + 48;
      *(_QWORD *)(v16 + 72) = v16 + 48;
      *(_QWORD *)(v16 + 80) = 0;
      v19 = (const __m128i *)*((_QWORD *)v15 + 7);
      if ( v19 )
      {
        v20 = sub_EF89A0(v19, v16 + 48);
        v21 = v20;
        do
        {
          v22 = v20;
          v20 = (__m128i *)v20[1].m128i_i64[0];
        }
        while ( v20 );
        *(_QWORD *)(v16 + 64) = v22;
        v23 = v21;
        do
        {
          v24 = v23;
          v23 = (__m128i *)v23[1].m128i_i64[1];
        }
        while ( v23 );
        *(_QWORD *)(v16 + 72) = v24;
        v25 = *((_QWORD *)v15 + 10);
        *(_QWORD *)(v16 + 56) = v21;
        *(_QWORD *)(v16 + 80) = v25;
      }
      v26 = *v15;
      *(_QWORD *)(v16 + 16) = 0;
      *(_QWORD *)(v16 + 24) = 0;
      *(_DWORD *)v16 = v26;
      *(_QWORD *)(v17 + 16) = v16;
      *(_QWORD *)(v16 + 8) = v17;
      v27 = *((_QWORD *)v15 + 3);
      if ( v27 )
        *(_QWORD *)(v16 + 24) = sub_EF8D70(v27, v16);
      v15 = (int *)*((_QWORD *)v15 + 2);
    }
    while ( v15 );
  }
  return v5;
}
