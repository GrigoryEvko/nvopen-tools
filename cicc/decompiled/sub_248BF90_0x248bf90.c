// Function: sub_248BF90
// Address: 0x248bf90
//
void __fastcall sub_248BF90(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // r12
  unsigned __int64 *v3; // rbx
  __int64 v4; // r14
  unsigned __int64 v5; // rax
  char *v6; // rdx
  unsigned __int64 *i; // rbx
  __m128i v8; // xmm0
  unsigned __int64 *v9; // r14
  unsigned __int64 *v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // r12
  unsigned __int64 v13; // rbx
  unsigned __int64 *v14; // r15
  unsigned __int64 *v15; // r13
  unsigned __int64 v16; // rdi
  __int64 v17; // r12
  unsigned __int64 v18; // r15
  unsigned __int64 *v19; // r14
  unsigned __int64 *v20; // rax
  unsigned __int64 *v21; // r13
  unsigned __int64 v22; // rdi
  __int64 v23; // rbx
  unsigned __int64 v24; // r15
  unsigned __int64 *v25; // r8
  unsigned __int64 *v26; // r14
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rdi
  __int64 v29; // r12
  unsigned __int64 v30; // rbx
  unsigned __int64 *v31; // r15
  unsigned __int64 v32; // r14
  unsigned __int64 v33; // r15
  unsigned __int64 v34; // r13
  unsigned __int64 *v35; // r8
  __m128i v36; // xmm2
  unsigned __int64 *v37; // r12
  unsigned __int64 v38; // rdi
  __int64 v39; // rbx
  unsigned __int64 v40; // r13
  unsigned __int64 *v41; // r8
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // r13
  unsigned __int64 v44; // r15
  unsigned __int64 *v45; // r8
  __m128i v46; // xmm6
  unsigned __int64 *v47; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v48; // [rsp+10h] [rbp-60h]
  unsigned int v49; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v51; // [rsp+30h] [rbp-40h]
  unsigned __int64 *v52; // [rsp+30h] [rbp-40h]
  unsigned __int64 *v53; // [rsp+38h] [rbp-38h]
  unsigned __int64 *v54; // [rsp+38h] [rbp-38h]
  unsigned __int64 *v55; // [rsp+38h] [rbp-38h]
  unsigned __int64 *v56; // [rsp+38h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = *(unsigned __int64 **)a1;
    v3 = (unsigned __int64 *)(a2 + 16);
    v51 = *(unsigned int *)(a1 + 8);
    v53 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v49 = *(_DWORD *)(a2 + 8);
      v4 = v49;
      if ( v49 > v51 )
      {
        if ( v49 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v37 = &v53[23 * v51];
          while ( v37 != v53 )
          {
            v38 = *(v37 - 23);
            v39 = *(v37 - 22);
            v37 -= 23;
            v40 = v38;
            if ( v39 != v38 )
            {
              do
              {
                v41 = *(unsigned __int64 **)(v40 + 8);
                if ( v41 )
                {
                  if ( (unsigned __int64 *)*v41 != v41 + 2 )
                  {
                    v52 = *(unsigned __int64 **)(v40 + 8);
                    j_j___libc_free_0(*v41);
                    v41 = v52;
                  }
                  j_j___libc_free_0((unsigned __int64)v41);
                }
                v40 += 32LL;
              }
              while ( v39 != v40 );
              v38 = *v37;
            }
            if ( v38 )
              j_j___libc_free_0(v38);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_C16ED0(a1, v49);
          v51 = 0;
          v3 = *(unsigned __int64 **)a2;
          v4 = *(unsigned int *)(a2 + 8);
          v53 = *(unsigned __int64 **)a1;
          v5 = *(_QWORD *)a2;
        }
        else
        {
          v5 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v51 *= 184LL;
            v47 = (unsigned __int64 *)((char *)v2 + v51);
            do
            {
              v32 = v2[1];
              v33 = *v2;
              v34 = v33;
              *v2 = *v3;
              v2[1] = v3[1];
              v2[2] = v3[2];
              *v3 = 0;
              v3[1] = 0;
              for ( v3[2] = 0; v32 != v34; v34 += 32LL )
              {
                v35 = *(unsigned __int64 **)(v34 + 8);
                if ( v35 )
                {
                  if ( (unsigned __int64 *)*v35 != v35 + 2 )
                  {
                    v55 = *(unsigned __int64 **)(v34 + 8);
                    j_j___libc_free_0(*v35);
                    v35 = v55;
                  }
                  j_j___libc_free_0((unsigned __int64)v35);
                }
              }
              if ( v33 )
                j_j___libc_free_0(v33);
              v36 = _mm_loadu_si128((const __m128i *)(v3 + 3));
              v2 += 23;
              v3 += 23;
              *((__m128i *)v2 - 10) = v36;
              *((__m128i *)v2 - 9) = _mm_loadu_si128((const __m128i *)v3 - 9);
              *((__m128i *)v2 - 8) = _mm_loadu_si128((const __m128i *)v3 - 8);
              *((__m128i *)v2 - 7) = _mm_loadu_si128((const __m128i *)v3 - 7);
              *((__m128i *)v2 - 6) = _mm_loadu_si128((const __m128i *)v3 - 6);
              *((__m128i *)v2 - 5) = _mm_loadu_si128((const __m128i *)v3 - 5);
              *((__m128i *)v2 - 4) = _mm_loadu_si128((const __m128i *)v3 - 4);
              *((__m128i *)v2 - 3) = _mm_loadu_si128((const __m128i *)v3 - 3);
              *((__m128i *)v2 - 2) = _mm_loadu_si128((const __m128i *)v3 - 2);
              *((__m128i *)v2 - 1) = _mm_loadu_si128((const __m128i *)v3 - 1);
            }
            while ( v2 != v47 );
            v3 = *(unsigned __int64 **)a2;
            v4 = *(unsigned int *)(a2 + 8);
            v53 = *(unsigned __int64 **)a1;
            v5 = *(_QWORD *)a2 + v51;
          }
        }
        v6 = (char *)v53 + v51;
        for ( i = &v3[23 * v4]; i != (unsigned __int64 *)v5; v6 += 184 )
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
          v5 += 184LL;
        }
        *(_DWORD *)(a1 + 8) = v49;
        v9 = *(unsigned __int64 **)a2;
        v10 = (unsigned __int64 *)(*(_QWORD *)a2 + 184LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v10 )
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
                v14 = *(unsigned __int64 **)(v13 + 8);
                if ( v14 )
                {
                  if ( (unsigned __int64 *)*v14 != v14 + 2 )
                    j_j___libc_free_0(*v14);
                  j_j___libc_free_0((unsigned __int64)v14);
                }
                v13 += 32LL;
              }
              while ( v12 != v13 );
              v11 = *v10;
            }
            if ( v11 )
              j_j___libc_free_0(v11);
          }
          while ( v9 != v10 );
        }
LABEL_21:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      v20 = *(unsigned __int64 **)a1;
      if ( v49 )
      {
        v48 = &v2[23 * v49];
        do
        {
          v42 = *v2;
          v43 = v2[1];
          v44 = v42;
          *v2 = *v3;
          v2[1] = v3[1];
          v2[2] = v3[2];
          *v3 = 0;
          v3[1] = 0;
          for ( v3[2] = 0; v43 != v44; v44 += 32LL )
          {
            v45 = *(unsigned __int64 **)(v44 + 8);
            if ( v45 )
            {
              if ( (unsigned __int64 *)*v45 != v45 + 2 )
              {
                v56 = *(unsigned __int64 **)(v44 + 8);
                j_j___libc_free_0(*v45);
                v45 = v56;
              }
              j_j___libc_free_0((unsigned __int64)v45);
            }
          }
          if ( v42 )
            j_j___libc_free_0(v42);
          v46 = _mm_loadu_si128((const __m128i *)(v3 + 3));
          v2 += 23;
          v3 += 23;
          *((__m128i *)v2 - 10) = v46;
          *((__m128i *)v2 - 9) = _mm_loadu_si128((const __m128i *)v3 - 9);
          *((__m128i *)v2 - 8) = _mm_loadu_si128((const __m128i *)v3 - 8);
          *((__m128i *)v2 - 7) = _mm_loadu_si128((const __m128i *)v3 - 7);
          *((__m128i *)v2 - 6) = _mm_loadu_si128((const __m128i *)v3 - 6);
          *((__m128i *)v2 - 5) = _mm_loadu_si128((const __m128i *)v3 - 5);
          *((__m128i *)v2 - 4) = _mm_loadu_si128((const __m128i *)v3 - 4);
          *((__m128i *)v2 - 3) = _mm_loadu_si128((const __m128i *)v3 - 3);
          *((__m128i *)v2 - 2) = _mm_loadu_si128((const __m128i *)v3 - 2);
          *((__m128i *)v2 - 1) = _mm_loadu_si128((const __m128i *)v3 - 1);
        }
        while ( v2 != v48 );
        v20 = *(unsigned __int64 **)a1;
        v51 = *(unsigned int *)(a1 + 8);
      }
      v21 = &v20[23 * v51];
      while ( v2 != v21 )
      {
        v22 = *(v21 - 23);
        v23 = *(v21 - 22);
        v21 -= 23;
        v24 = v22;
        if ( v23 != v22 )
        {
          do
          {
            v25 = *(unsigned __int64 **)(v24 + 8);
            if ( v25 )
            {
              if ( (unsigned __int64 *)*v25 != v25 + 2 )
              {
                v54 = *(unsigned __int64 **)(v24 + 8);
                j_j___libc_free_0(*v25);
                v25 = v54;
              }
              j_j___libc_free_0((unsigned __int64)v25);
            }
            v24 += 32LL;
          }
          while ( v23 != v24 );
          v22 = *v21;
        }
        if ( v22 )
          j_j___libc_free_0(v22);
      }
      *(_DWORD *)(a1 + 8) = v49;
      v26 = *(unsigned __int64 **)a2;
      v27 = (unsigned __int64 *)(*(_QWORD *)a2 + 184LL * *(unsigned int *)(a2 + 8));
      if ( *(unsigned __int64 **)a2 == v27 )
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
            v31 = *(unsigned __int64 **)(v30 + 8);
            if ( v31 )
            {
              if ( (unsigned __int64 *)*v31 != v31 + 2 )
                j_j___libc_free_0(*v31);
              j_j___libc_free_0((unsigned __int64)v31);
            }
            v30 += 32LL;
          }
          while ( v29 != v30 );
          v28 = *v27;
        }
        if ( v28 )
          j_j___libc_free_0(v28);
      }
      while ( v26 != v27 );
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v15 = &v2[23 * *(unsigned int *)(a1 + 8)];
      if ( v15 != v2 )
      {
        do
        {
          v16 = *(v15 - 23);
          v17 = *(v15 - 22);
          v15 -= 23;
          v18 = v16;
          if ( v17 != v16 )
          {
            do
            {
              v19 = *(unsigned __int64 **)(v18 + 8);
              if ( v19 )
              {
                if ( (unsigned __int64 *)*v19 != v19 + 2 )
                  j_j___libc_free_0(*v19);
                j_j___libc_free_0((unsigned __int64)v19);
              }
              v18 += 32LL;
            }
            while ( v17 != v18 );
            v16 = *v15;
          }
          if ( v16 )
            j_j___libc_free_0(v16);
        }
        while ( v15 != v53 );
        v2 = *(unsigned __int64 **)a1;
      }
      if ( v2 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v2);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v3;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
