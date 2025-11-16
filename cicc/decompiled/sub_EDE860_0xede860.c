// Function: sub_EDE860
// Address: 0xede860
//
void __fastcall sub_EDE860(__int64 a1, __int64 a2)
{
  __m128i *v2; // r12
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int8 *v6; // rdx
  __int64 i; // rbx
  __m128i v8; // xmm0
  __int64 *v9; // r14
  __int64 *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // rbx
  _QWORD *v14; // r15
  __m128i *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // r12
  __int64 v18; // r15
  _QWORD *v19; // r14
  __m128i *v20; // rax
  __m128i *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // r15
  _QWORD *v25; // r8
  __int64 *v26; // r14
  __int64 *v27; // r13
  __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rbx
  _QWORD *v31; // r15
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // r13
  _QWORD *v35; // r8
  __m128i v36; // xmm2
  __m128i *v37; // r12
  __int64 v38; // rdi
  __int64 v39; // rbx
  __int64 v40; // r13
  _QWORD *v41; // r8
  __int64 v42; // r14
  __int64 v43; // r13
  __int64 v44; // r15
  _QWORD *v45; // r8
  __m128i v46; // xmm6
  __m128i *v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __m128i *v49; // [rsp+10h] [rbp-60h]
  unsigned int v50; // [rsp+1Ch] [rbp-54h]
  __int64 v52; // [rsp+28h] [rbp-48h]
  unsigned __int64 v53; // [rsp+30h] [rbp-40h]
  _QWORD *v54; // [rsp+30h] [rbp-40h]
  __int64 v55; // [rsp+30h] [rbp-40h]
  __m128i *v56; // [rsp+38h] [rbp-38h]
  _QWORD *v57; // [rsp+38h] [rbp-38h]
  _QWORD *v58; // [rsp+38h] [rbp-38h]
  _QWORD *v59; // [rsp+38h] [rbp-38h]

  v52 = a2;
  if ( a1 != a2 )
  {
    v2 = *(__m128i **)a1;
    v3 = a2 + 16;
    v53 = *(unsigned int *)(a1 + 8);
    v56 = *(__m128i **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v50 = *(_DWORD *)(a2 + 8);
      v4 = v50;
      if ( v50 > v53 )
      {
        if ( v50 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v37 = (__m128i *)((char *)v56 + 184 * v53);
          while ( v37 != v56 )
          {
            v38 = v37[-12].m128i_i64[1];
            v39 = v37[-11].m128i_i64[0];
            v37 = (__m128i *)((char *)v37 - 184);
            v40 = v38;
            if ( v39 != v38 )
            {
              do
              {
                v41 = *(_QWORD **)(v40 + 8);
                if ( v41 )
                {
                  if ( (_QWORD *)*v41 != v41 + 2 )
                  {
                    v54 = *(_QWORD **)(v40 + 8);
                    j_j___libc_free_0(*v41, v41[2] + 1LL);
                    v41 = v54;
                  }
                  j_j___libc_free_0(v41, 32);
                }
                v40 += 32;
              }
              while ( v39 != v40 );
              v38 = v37->m128i_i64[0];
            }
            if ( v38 )
              j_j___libc_free_0(v38, v37[1].m128i_i64[0] - v38);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_C16ED0(a1, v50);
          v53 = 0;
          v3 = *(_QWORD *)a2;
          v4 = *(unsigned int *)(a2 + 8);
          v56 = *(__m128i **)a1;
          v5 = *(_QWORD *)a2;
        }
        else
        {
          v5 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v53 *= 184LL;
            v47 = (__m128i *)((char *)v2 + v53);
            do
            {
              v32 = v2->m128i_i64[1];
              v48 = v2[1].m128i_i64[0];
              v33 = v2->m128i_i64[0];
              v34 = v33;
              v2->m128i_i64[0] = *(_QWORD *)v3;
              v2->m128i_i64[1] = *(_QWORD *)(v3 + 8);
              v2[1].m128i_i64[0] = *(_QWORD *)(v3 + 16);
              *(_QWORD *)v3 = 0;
              *(_QWORD *)(v3 + 8) = 0;
              for ( *(_QWORD *)(v3 + 16) = 0; v32 != v34; v34 += 32 )
              {
                v35 = *(_QWORD **)(v34 + 8);
                if ( v35 )
                {
                  if ( (_QWORD *)*v35 != v35 + 2 )
                  {
                    v58 = *(_QWORD **)(v34 + 8);
                    j_j___libc_free_0(*v35, v35[2] + 1LL);
                    v35 = v58;
                  }
                  j_j___libc_free_0(v35, 32);
                }
              }
              if ( v33 )
                j_j___libc_free_0(v33, v48 - v33);
              v36 = _mm_loadu_si128((const __m128i *)(v3 + 24));
              v2 = (__m128i *)((char *)v2 + 184);
              v3 += 184;
              v2[-10] = v36;
              v2[-9] = _mm_loadu_si128((const __m128i *)(v3 - 144));
              v2[-8] = _mm_loadu_si128((const __m128i *)(v3 - 128));
              v2[-7] = _mm_loadu_si128((const __m128i *)(v3 - 112));
              v2[-6] = _mm_loadu_si128((const __m128i *)(v3 - 96));
              v2[-5] = _mm_loadu_si128((const __m128i *)(v3 - 80));
              v2[-4] = _mm_loadu_si128((const __m128i *)(v3 - 64));
              v2[-3] = _mm_loadu_si128((const __m128i *)(v3 - 48));
              v2[-2] = _mm_loadu_si128((const __m128i *)(v3 - 32));
              v2[-1] = _mm_loadu_si128((const __m128i *)(v3 - 16));
            }
            while ( v2 != v47 );
            v3 = *(_QWORD *)a2;
            v4 = *(unsigned int *)(a2 + 8);
            v56 = *(__m128i **)a1;
            v5 = *(_QWORD *)a2 + v53;
          }
        }
        v6 = &v56->m128i_i8[v53];
        for ( i = 184 * v4 + v3; i != v5; v6 += 184 )
        {
          if ( v6 )
          {
            *(_QWORD *)v6 = *(_QWORD *)v5;
            *((_QWORD *)v6 + 1) = *(_QWORD *)(v5 + 8);
            *((_QWORD *)v6 + 2) = *(_QWORD *)(v5 + 16);
            v8 = _mm_loadu_si128((const __m128i *)(v5 + 24));
            *(_QWORD *)(v5 + 16) = 0;
            *(_QWORD *)(v5 + 8) = 0;
            *(_QWORD *)v5 = 0;
            *(__m128i *)(v6 + 24) = v8;
            *(__m128i *)(v6 + 40) = _mm_loadu_si128((const __m128i *)(v5 + 40));
            *(__m128i *)(v6 + 56) = _mm_loadu_si128((const __m128i *)(v5 + 56));
            *(__m128i *)(v6 + 72) = _mm_loadu_si128((const __m128i *)(v5 + 72));
            *(__m128i *)(v6 + 88) = _mm_loadu_si128((const __m128i *)(v5 + 88));
            *(__m128i *)(v6 + 104) = _mm_loadu_si128((const __m128i *)(v5 + 104));
            *(__m128i *)(v6 + 120) = _mm_loadu_si128((const __m128i *)(v5 + 120));
            *(__m128i *)(v6 + 136) = _mm_loadu_si128((const __m128i *)(v5 + 136));
            *(__m128i *)(v6 + 152) = _mm_loadu_si128((const __m128i *)(v5 + 152));
            *(__m128i *)(v6 + 168) = _mm_loadu_si128((const __m128i *)(v5 + 168));
          }
          v5 += 184;
        }
        *(_DWORD *)(a1 + 8) = v50;
        v9 = *(__int64 **)a2;
        v10 = (__int64 *)(*(_QWORD *)a2 + 184LL * *(unsigned int *)(a2 + 8));
        if ( *(__int64 **)a2 != v10 )
        {
          do
          {
            v11 = *(v10 - 23);
            v12 = *(v10 - 22);
            v10 -= 23;
            v13 = v11;
            if ( v12 != v11 )
            {
              do
              {
                v14 = *(_QWORD **)(v13 + 8);
                if ( v14 )
                {
                  if ( (_QWORD *)*v14 != v14 + 2 )
                    j_j___libc_free_0(*v14, v14[2] + 1LL);
                  j_j___libc_free_0(v14, 32);
                }
                v13 += 32;
              }
              while ( v12 != v13 );
              v11 = *v10;
            }
            if ( v11 )
              j_j___libc_free_0(v11, v10[2] - v11);
          }
          while ( v9 != v10 );
        }
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      v20 = *(__m128i **)a1;
      if ( v50 )
      {
        v49 = (__m128i *)((char *)v2 + 184 * v50);
        do
        {
          v42 = v2->m128i_i64[0];
          v43 = v2->m128i_i64[1];
          v55 = v2[1].m128i_i64[0];
          v44 = v42;
          v2->m128i_i64[0] = *(_QWORD *)v3;
          v2->m128i_i64[1] = *(_QWORD *)(v3 + 8);
          v2[1].m128i_i64[0] = *(_QWORD *)(v3 + 16);
          *(_QWORD *)v3 = 0;
          *(_QWORD *)(v3 + 8) = 0;
          for ( *(_QWORD *)(v3 + 16) = 0; v43 != v44; v44 += 32 )
          {
            v45 = *(_QWORD **)(v44 + 8);
            if ( v45 )
            {
              if ( (_QWORD *)*v45 != v45 + 2 )
              {
                v59 = *(_QWORD **)(v44 + 8);
                j_j___libc_free_0(*v45, v45[2] + 1LL);
                v45 = v59;
              }
              j_j___libc_free_0(v45, 32);
            }
          }
          if ( v42 )
            j_j___libc_free_0(v42, v55 - v42);
          v46 = _mm_loadu_si128((const __m128i *)(v3 + 24));
          v2 = (__m128i *)((char *)v2 + 184);
          v3 += 184;
          v2[-10] = v46;
          v2[-9] = _mm_loadu_si128((const __m128i *)(v3 - 144));
          v2[-8] = _mm_loadu_si128((const __m128i *)(v3 - 128));
          v2[-7] = _mm_loadu_si128((const __m128i *)(v3 - 112));
          v2[-6] = _mm_loadu_si128((const __m128i *)(v3 - 96));
          v2[-5] = _mm_loadu_si128((const __m128i *)(v3 - 80));
          v2[-4] = _mm_loadu_si128((const __m128i *)(v3 - 64));
          v2[-3] = _mm_loadu_si128((const __m128i *)(v3 - 48));
          v2[-2] = _mm_loadu_si128((const __m128i *)(v3 - 32));
          v2[-1] = _mm_loadu_si128((const __m128i *)(v3 - 16));
        }
        while ( v2 != v49 );
        v20 = *(__m128i **)a1;
        v53 = *(unsigned int *)(a1 + 8);
      }
      v21 = (__m128i *)((char *)v20 + 184 * v53);
      while ( v2 != v21 )
      {
        v22 = v21[-12].m128i_i64[1];
        v23 = v21[-11].m128i_i64[0];
        v21 = (__m128i *)((char *)v21 - 184);
        v24 = v22;
        if ( v23 != v22 )
        {
          do
          {
            v25 = *(_QWORD **)(v24 + 8);
            if ( v25 )
            {
              if ( (_QWORD *)*v25 != v25 + 2 )
              {
                v57 = *(_QWORD **)(v24 + 8);
                j_j___libc_free_0(*v25, v25[2] + 1LL);
                v25 = v57;
              }
              j_j___libc_free_0(v25, 32);
            }
            v24 += 32;
          }
          while ( v23 != v24 );
          v22 = v21->m128i_i64[0];
        }
        if ( v22 )
          j_j___libc_free_0(v22, v21[1].m128i_i64[0] - v22);
      }
      *(_DWORD *)(a1 + 8) = v50;
      v26 = *(__int64 **)a2;
      v27 = (__int64 *)(*(_QWORD *)a2 + 184LL * *(unsigned int *)(a2 + 8));
      if ( *(__int64 **)a2 == v27 )
        goto LABEL_21;
      do
      {
        v28 = *(v27 - 23);
        v29 = *(v27 - 22);
        v27 -= 23;
        v30 = v28;
        if ( v29 != v28 )
        {
          do
          {
            v31 = *(_QWORD **)(v30 + 8);
            if ( v31 )
            {
              if ( (_QWORD *)*v31 != v31 + 2 )
                j_j___libc_free_0(*v31, v31[2] + 1LL);
              j_j___libc_free_0(v31, 32);
            }
            v30 += 32;
          }
          while ( v29 != v30 );
          v28 = *v27;
        }
        if ( v28 )
          j_j___libc_free_0(v28, v27[2] - v28);
      }
      while ( v26 != v27 );
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v15 = (__m128i *)((char *)v2 + 184 * *(unsigned int *)(a1 + 8));
      if ( v15 != v2 )
      {
        do
        {
          v16 = v15[-12].m128i_i64[1];
          v17 = v15[-11].m128i_i64[0];
          v15 = (__m128i *)((char *)v15 - 184);
          v18 = v16;
          if ( v17 != v16 )
          {
            do
            {
              v19 = *(_QWORD **)(v18 + 8);
              if ( v19 )
              {
                if ( (_QWORD *)*v19 != v19 + 2 )
                  j_j___libc_free_0(*v19, v19[2] + 1LL);
                a2 = 32;
                j_j___libc_free_0(v19, 32);
              }
              v18 += 32;
            }
            while ( v17 != v18 );
            v16 = v15->m128i_i64[0];
          }
          if ( v16 )
          {
            a2 = v15[1].m128i_i64[0] - v16;
            j_j___libc_free_0(v16, a2);
          }
        }
        while ( v15 != v56 );
        v2 = *(__m128i **)a1;
      }
      if ( v2 != (__m128i *)(a1 + 16) )
        _libc_free(v2, a2);
      *(_QWORD *)a1 = *(_QWORD *)v52;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(v52 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(v52 + 12);
      *(_QWORD *)v52 = v3;
      *(_QWORD *)(v52 + 8) = 0;
    }
  }
}
