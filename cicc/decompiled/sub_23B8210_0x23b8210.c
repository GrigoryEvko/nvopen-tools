// Function: sub_23B8210
// Address: 0x23b8210
//
unsigned __int64 __fastcall sub_23B8210(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // r10
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  __m128i *v13; // r12
  const __m128i *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __m128i *v18; // rdi
  __int64 v19; // rax
  unsigned __int64 result; // rax
  const __m128i *v21; // rdx
  __int8 *v22; // r14
  __int64 v23; // rbx
  __int8 *v24; // r13
  unsigned __int64 v25; // rdi
  int v26; // r12d
  unsigned __int64 v27; // rdi
  int v28; // r12d
  __int64 v29; // rbx
  __int8 *v30; // r13
  __m128i *v31; // r14
  unsigned __int64 v32; // rdi
  int v33; // r12d
  __int64 v34; // rdx
  __m128i *v35; // r14
  unsigned __int64 v36; // rdi
  int v37; // r12d
  __int64 v38; // rdx
  char v39; // [rsp+0h] [rbp-60h]
  __m128i *v40; // [rsp+0h] [rbp-60h]
  __m128i *v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43[7]; // [rsp+28h] [rbp-38h] BYREF

  v8 = *(_QWORD *)a1;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = v9 + 1;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = 32 * v9;
  v13 = (__m128i *)(*(_QWORD *)a1 + 32 * v9);
  if ( v13 == a2 )
  {
    if ( v10 > v11 )
    {
      v29 = a1 + 16;
      if ( v8 > (unsigned __int64)a3 || v13 <= a3 )
      {
        v35 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v10, 0x20u, v43, a6);
        sub_BC3FC0(a1, v35);
        v36 = *(_QWORD *)a1;
        v37 = v43[0];
        if ( *(_QWORD *)a1 != v29 )
          _libc_free(v36);
        v38 = *(unsigned int *)(a1 + 8);
        *(_QWORD *)a1 = v35;
        *(_DWORD *)(a1 + 12) = v37;
        LODWORD(v9) = v38;
        a2 = &v35[2 * v38];
      }
      else
      {
        v30 = &a3->m128i_i8[-v8];
        v31 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v10, 0x20u, v43, a6);
        sub_BC3FC0(a1, v31);
        v32 = *(_QWORD *)a1;
        v33 = v43[0];
        if ( *(_QWORD *)a1 != v29 )
          _libc_free(v32);
        v34 = *(unsigned int *)(a1 + 8);
        *(_QWORD *)a1 = v31;
        a3 = (const __m128i *)&v30[(_QWORD)v31];
        LODWORD(v9) = v34;
        *(_DWORD *)(a1 + 12) = v33;
        a2 = &v31[2 * v34];
      }
    }
    if ( a2 )
    {
      sub_23B8170(a2, a3);
      LODWORD(v9) = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v9 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    if ( v10 > v11 )
    {
      v22 = &a2->m128i_i8[-v8];
      v23 = a1 + 16;
      if ( v8 > (unsigned __int64)a3 || v13 <= a3 )
      {
        v40 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v10, 0x20u, v43, a6);
        sub_BC3FC0(a1, v40);
        v27 = *(_QWORD *)a1;
        v28 = v43[0];
        v8 = (unsigned __int64)v40;
        if ( *(_QWORD *)a1 != v23 )
        {
          _libc_free(v27);
          v8 = (unsigned __int64)v40;
        }
        *(_QWORD *)a1 = v8;
        *(_DWORD *)(a1 + 12) = v28;
      }
      else
      {
        v24 = &a3->m128i_i8[-v8];
        v42 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v10, 0x20u, v43, a6);
        sub_BC3FC0(a1, v42);
        v25 = *(_QWORD *)a1;
        v26 = v43[0];
        v8 = (unsigned __int64)v42;
        if ( *(_QWORD *)a1 != v23 )
        {
          _libc_free(v25);
          v8 = (unsigned __int64)v42;
        }
        *(_QWORD *)a1 = v8;
        *(_DWORD *)(a1 + 12) = v26;
        a3 = (const __m128i *)&v24[v8];
      }
      a2 = (__m128i *)&v22[v8];
      v9 = *(unsigned int *)(a1 + 8);
      v12 = 32 * v9;
      v13 = (__m128i *)(v8 + 32 * v9);
    }
    v14 = (const __m128i *)(v8 + v12 - 32);
    if ( v13 )
    {
      sub_23B8170(v13, (const __m128i *)(v8 + v12 - 32));
      v8 = *(_QWORD *)a1;
      v9 = *(unsigned int *)(a1 + 8);
      v13 = (__m128i *)(*(_QWORD *)a1 + 32 * v9);
      v14 = v13 - 2;
    }
    v15 = ((char *)v14 - (char *)a2) >> 5;
    if ( (char *)v14 - (char *)a2 > 0 )
    {
      do
      {
        v14 -= 2;
        v13 -= 2;
        if ( v14 != v13 )
        {
          v16 = v13[1].m128i_i64[1];
          if ( (v16 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v17 = (v16 >> 1) & 1;
            if ( (v16 & 4) != 0 )
            {
              v18 = v13;
              if ( !(_BYTE)v17 )
                v18 = (__m128i *)v13->m128i_i64[0];
              v39 = (v16 >> 1) & 1;
              (*(void (__fastcall **)(__m128i *))((v13[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(v18);
              LOBYTE(v17) = v39;
            }
            if ( !(_BYTE)v17 )
              sub_C7D6A0(v13->m128i_i64[0], v13->m128i_i64[1], v13[1].m128i_i64[0]);
          }
          v13[1].m128i_i64[1] = 0;
          v19 = v14[1].m128i_i64[1];
          v13[1].m128i_i64[1] = v19;
          if ( (v14[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            if ( (v19 & 2) != 0 && (v19 & 4) != 0 )
            {
              (*(void (__fastcall **)(__m128i *, const __m128i *))((v19 & 0xFFFFFFFFFFFFFFF8LL) + 8))(v13, v14);
              (*(void (__fastcall **)(const __m128i *))((v13[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(v14);
            }
            else
            {
              *v13 = _mm_loadu_si128(v14);
              v13[1].m128i_i64[0] = v14[1].m128i_i64[0];
            }
            v14[1].m128i_i64[1] = 0;
          }
        }
        --v15;
      }
      while ( v15 );
      LODWORD(v9) = *(_DWORD *)(a1 + 8);
      v8 = *(_QWORD *)a1;
    }
    result = (unsigned int)(v9 + 1);
    *(_DWORD *)(a1 + 8) = result;
    if ( a2 <= a3 )
    {
      result = v8 + 32 * result;
      v21 = a3 + 2;
      if ( (unsigned __int64)a3 >= result )
        v21 = a3;
      a3 = v21;
    }
    if ( a2 != a3 )
    {
      sub_23B7140(a2->m128i_i64);
      a2[1].m128i_i64[1] = 0;
      result = a3[1].m128i_u64[1];
      a2[1].m128i_i64[1] = result;
      if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (result & 2) != 0 && (result & 4) != 0 )
        {
          (*(void (__fastcall **)(__m128i *, const __m128i *))((result & 0xFFFFFFFFFFFFFFF8LL) + 8))(a2, a3);
          (*(void (__fastcall **)(const __m128i *))((a2[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(a3);
        }
        else
        {
          *a2 = _mm_loadu_si128(a3);
          a2[1].m128i_i64[0] = a3[1].m128i_i64[0];
        }
        result = (unsigned __int64)a3;
        a3[1].m128i_i64[1] = 0;
      }
    }
  }
  return result;
}
