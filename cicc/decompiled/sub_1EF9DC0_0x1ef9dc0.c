// Function: sub_1EF9DC0
// Address: 0x1ef9dc0
//
void __fastcall sub_1EF9DC0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rcx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rbx
  __m128i v11; // xmm1
  int v12; // eax
  __int64 v13; // r14
  const __m128i *v14; // rbx
  __m128i *v15; // r15
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  unsigned int v19; // ebx
  __int64 v20; // r13
  unsigned __int64 v21; // rax
  void *v22; // rbx
  __int64 v23; // r12
  int v24; // ecx
  __int64 v25; // rdx
  int v26; // ecx
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rbx
  size_t v30; // r14
  void *v31; // r15
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v34 = a3;
  v7 = *(unsigned int *)(a1 + 12);
  LODWORD(a3) = v6;
  v8 = 32 * v6;
  v9 = v5 + 32 * v6;
  if ( v9 == a2 )
  {
    if ( (unsigned int)v6 >= (unsigned int)v7 )
    {
      sub_1EF9C40(a1, 0);
      a3 = *(unsigned int *)(a1 + 8);
      a2 = *(_QWORD *)a1 + 32 * a3;
    }
    if ( a2 )
    {
      v27 = *(_QWORD *)v34;
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)a2 = v27;
      v28 = *(_DWORD *)(v34 + 24);
      *(_DWORD *)(a2 + 24) = v28;
      if ( v28 )
      {
        v29 = (unsigned int)(v28 + 63) >> 6;
        v30 = 8 * v29;
        v31 = (void *)malloc(8 * v29);
        if ( !v31 )
        {
          if ( v30 || (v33 = malloc(1u)) == 0 )
            sub_16BD1C0("Allocation failed", 1u);
          else
            v31 = (void *)v33;
        }
        *(_QWORD *)(a2 + 8) = v31;
        *(_QWORD *)(a2 + 16) = v29;
        memcpy(v31, *(const void **)(v34 + 8), v30);
      }
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = a3 + 1;
  }
  else
  {
    if ( v6 >= v7 )
    {
      v23 = a2 - v5;
      sub_1EF9C40(a1, 0);
      v5 = *(_QWORD *)a1;
      a3 = *(unsigned int *)(a1 + 8);
      v8 = 32 * a3;
      a2 = *(_QWORD *)a1 + v23;
      v9 = *(_QWORD *)a1 + 32 * a3;
    }
    v10 = v5 + v8 - 32;
    if ( v9 )
    {
      *(_QWORD *)v9 = *(_QWORD *)v10;
      v11 = _mm_loadu_si128((const __m128i *)(v10 + 8));
      v12 = *(_DWORD *)(v10 + 24);
      *(_QWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 0;
      *(_DWORD *)(v9 + 24) = v12;
      *(_DWORD *)(v10 + 24) = 0;
      *(__m128i *)(v9 + 8) = v11;
      a3 = *(unsigned int *)(a1 + 8);
      v9 = *(_QWORD *)a1 + 32 * a3;
      v10 = v9 - 32;
    }
    v13 = (__int64)(v10 - a2) >> 5;
    if ( (__int64)(v10 - a2) > 0 )
    {
      v14 = (const __m128i *)(v10 - 24);
      v15 = (__m128i *)(v9 - 24);
      do
      {
        v15[-1].m128i_i32[2] = v14[-1].m128i_i32[2];
        v15[-1].m128i_i32[3] = v14[-1].m128i_i32[3];
        if ( v15 != v14 )
        {
          _libc_free(v15->m128i_i64[0]);
          *v15 = _mm_loadu_si128(v14);
          v15[1].m128i_i32[0] = v14[1].m128i_i32[0];
          v14->m128i_i64[0] = 0;
          v14->m128i_i64[1] = 0;
          v14[1].m128i_i32[0] = 0;
        }
        v14 -= 2;
        v15 -= 2;
        --v13;
      }
      while ( v13 );
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
    }
    v16 = (unsigned int)(a3 + 1);
    *(_DWORD *)(a1 + 8) = v16;
    if ( v34 >= a2 )
    {
      v21 = v34 + 32;
      if ( v34 >= *(_QWORD *)a1 + 32 * v16 )
        v21 = v34;
      v34 = v21;
    }
    *(_DWORD *)a2 = *(_DWORD *)v34;
    *(_DWORD *)(a2 + 4) = *(_DWORD *)(v34 + 4);
    if ( v34 + 8 != a2 + 8 )
    {
      v17 = *(unsigned int *)(v34 + 24);
      v18 = *(_QWORD *)(a2 + 16);
      *(_DWORD *)(a2 + 24) = v17;
      v19 = (unsigned int)(v17 + 63) >> 6;
      v20 = v19;
      if ( v17 <= v18 << 6 )
      {
        if ( (_DWORD)v17 )
        {
          memcpy(*(void **)(a2 + 8), *(const void **)(v34 + 8), 8LL * v19);
          v24 = *(_DWORD *)(a2 + 24);
          v18 = *(_QWORD *)(a2 + 16);
          v19 = (unsigned int)(v24 + 63) >> 6;
          v20 = v19;
          if ( v18 <= v19 )
          {
LABEL_27:
            v26 = v24 & 0x3F;
            if ( v26 )
              *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL * (v19 - 1)) &= ~(-1LL << v26);
            return;
          }
        }
        else if ( v18 <= v19 )
        {
          return;
        }
        v25 = v18 - v20;
        if ( v25 )
          memset((void *)(*(_QWORD *)(a2 + 8) + 8 * v20), 0, 8 * v25);
        v24 = *(_DWORD *)(a2 + 24);
        goto LABEL_27;
      }
      v22 = (void *)malloc(8LL * v19);
      if ( !v22 )
      {
        if ( 8 * v20 || (v32 = malloc(1u)) == 0 )
          sub_16BD1C0("Allocation failed", 1u);
        else
          v22 = (void *)v32;
      }
      memcpy(v22, *(const void **)(v34 + 8), 8 * v20);
      _libc_free(*(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 8) = v22;
      *(_QWORD *)(a2 + 16) = v20;
    }
  }
}
