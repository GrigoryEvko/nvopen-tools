// Function: sub_C17210
// Address: 0xc17210
//
__int64 __fastcall sub_C17210(_QWORD *a1, __int64 *a2, void (__fastcall *a3)(__int64 *, __int64, _QWORD), __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // r13
  __int64 v9; // rbx
  int v10; // edi
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  char *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rsi
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __m128i v22; // xmm7
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __int64 v27; // r15
  _QWORD *v28; // r12
  __int64 v29; // rsi
  _QWORD *v30; // r13
  __int64 v31; // rcx
  __int64 v32; // rsi
  int v33; // eax
  __int64 v34; // rsi
  __int64 *v35; // rdx
  _QWORD *v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r15
  _QWORD *v42; // r14
  unsigned __int64 v44; // rbx
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rsi
  __int64 v50; // [rsp+10h] [rbp-140h]
  __int64 v51; // [rsp+18h] [rbp-138h]
  __int64 v52; // [rsp+18h] [rbp-138h]
  __int64 v55; // [rsp+30h] [rbp-120h]
  _QWORD *i; // [rsp+30h] [rbp-120h]
  __int64 v57; // [rsp+40h] [rbp-110h] BYREF
  __int64 v58; // [rsp+48h] [rbp-108h]
  __int64 v59; // [rsp+50h] [rbp-100h]
  __int64 v60; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v61; // [rsp+68h] [rbp-E8h]
  __int64 v62; // [rsp+70h] [rbp-E0h]
  __m128i v63; // [rsp+78h] [rbp-D8h]
  __m128i v64; // [rsp+88h] [rbp-C8h]
  __m128i v65; // [rsp+98h] [rbp-B8h]
  __m128i v66; // [rsp+A8h] [rbp-A8h]
  __m128i v67; // [rsp+B8h] [rbp-98h]
  __m128i v68; // [rsp+C8h] [rbp-88h]
  __m128i v69; // [rsp+D8h] [rbp-78h]
  __m128i v70; // [rsp+E8h] [rbp-68h]
  __m128i v71; // [rsp+F8h] [rbp-58h]
  __m128i v72; // [rsp+108h] [rbp-48h]

  v4 = (__int64)a1;
  *a1 = a1 + 2;
  a1[1] = 0x100000000LL;
  a1[25] = a1 + 27;
  a1[26] = 0x200000000LL;
  v5 = *((unsigned int *)a2 + 2);
  if ( (unsigned int)v5 > 1 )
  {
    sub_C16ED0((__int64)a1, v5);
    v5 = *((unsigned int *)a2 + 2);
  }
  v6 = *a2;
  v50 = *a2 + 168 * v5;
  if ( *a2 != v50 )
  {
    do
    {
      v63 = _mm_loadu_si128((const __m128i *)(v6 + 8));
      v64 = _mm_loadu_si128((const __m128i *)(v6 + 24));
      v65 = _mm_loadu_si128((const __m128i *)(v6 + 40));
      v66 = _mm_loadu_si128((const __m128i *)(v6 + 56));
      v67 = _mm_loadu_si128((const __m128i *)(v6 + 72));
      v68 = _mm_loadu_si128((const __m128i *)(v6 + 88));
      v69 = _mm_loadu_si128((const __m128i *)(v6 + 104));
      v70 = _mm_loadu_si128((const __m128i *)(v6 + 120));
      v71 = _mm_loadu_si128((const __m128i *)(v6 + 136));
      v72 = _mm_loadu_si128((const __m128i *)(v6 + 152));
      a3(&v57, a4, *(_QWORD *)v6);
      v7 = *(unsigned int *)(v4 + 8);
      v8 = v57;
      v9 = v58;
      v55 = v59;
      v10 = *(_DWORD *)(v4 + 8);
      v62 = v59;
      v11 = *(unsigned int *)(v4 + 12);
      v60 = v57;
      v61 = v58;
      if ( v7 + 1 > v11 )
      {
        v45 = *(_QWORD *)v4;
        if ( *(_QWORD *)v4 > (unsigned __int64)&v60 )
        {
          v48 = v7 + 1;
          v47 = v4;
LABEL_45:
          sub_C16ED0(v47, v48);
          v7 = *(unsigned int *)(v4 + 8);
          v12 = *(_QWORD *)v4;
          v13 = (char *)&v60;
          v10 = *(_DWORD *)(v4 + 8);
          goto LABEL_6;
        }
        v46 = 184 * v7;
        v47 = v4;
        v48 = v7 + 1;
        if ( (unsigned __int64)&v60 >= v45 + v46 )
          goto LABEL_45;
        sub_C16ED0(v4, v48);
        v12 = *(_QWORD *)v4;
        v7 = *(unsigned int *)(v4 + 8);
        v13 = (char *)&v60 + *(_QWORD *)v4 - v45;
        v10 = *(_DWORD *)(v4 + 8);
      }
      else
      {
        v12 = *(_QWORD *)v4;
        v13 = (char *)&v60;
      }
LABEL_6:
      v14 = 184 * v7 + v12;
      if ( v14 )
      {
        v15 = *(_QWORD *)v13;
        *(_QWORD *)v13 = 0;
        *(_QWORD *)v14 = v15;
        v16 = *((_QWORD *)v13 + 1);
        *((_QWORD *)v13 + 1) = 0;
        *(_QWORD *)(v14 + 8) = v16;
        v17 = *((_QWORD *)v13 + 2);
        *((_QWORD *)v13 + 2) = 0;
        *(_QWORD *)(v14 + 16) = v17;
        v18 = _mm_loadu_si128((const __m128i *)(v13 + 40));
        v19 = _mm_loadu_si128((const __m128i *)(v13 + 56));
        v20 = _mm_loadu_si128((const __m128i *)(v13 + 72));
        *(__m128i *)(v14 + 24) = _mm_loadu_si128((const __m128i *)(v13 + 24));
        v21 = _mm_loadu_si128((const __m128i *)(v13 + 88));
        *(__m128i *)(v14 + 40) = v18;
        v22 = _mm_loadu_si128((const __m128i *)(v13 + 104));
        v23 = _mm_loadu_si128((const __m128i *)(v13 + 120));
        *(__m128i *)(v14 + 56) = v19;
        v24 = _mm_loadu_si128((const __m128i *)(v13 + 136));
        v25 = _mm_loadu_si128((const __m128i *)(v13 + 152));
        *(__m128i *)(v14 + 72) = v20;
        v26 = _mm_loadu_si128((const __m128i *)(v13 + 168));
        *(__m128i *)(v14 + 88) = v21;
        *(__m128i *)(v14 + 104) = v22;
        *(__m128i *)(v14 + 120) = v23;
        *(__m128i *)(v14 + 136) = v24;
        *(__m128i *)(v14 + 152) = v25;
        *(__m128i *)(v14 + 168) = v26;
        v9 = v61;
        v8 = v60;
        v10 = *(_DWORD *)(v4 + 8);
        v55 = v62;
      }
      v27 = v8;
      *(_DWORD *)(v4 + 8) = v10 + 1;
      if ( v9 != v8 )
      {
        v51 = v4;
        do
        {
          v28 = *(_QWORD **)(v27 + 8);
          if ( v28 )
          {
            if ( (_QWORD *)*v28 != v28 + 2 )
              j_j___libc_free_0(*v28, v28[2] + 1LL);
            j_j___libc_free_0(v28, 32);
          }
          v27 += 32;
        }
        while ( v9 != v27 );
        v4 = v51;
      }
      if ( v8 )
        j_j___libc_free_0(v8, v55 - v8);
      v6 += 168;
    }
    while ( v50 != v6 );
  }
  v29 = *((unsigned int *)a2 + 48);
  if ( *(_DWORD *)(v4 + 212) < (unsigned int)v29 )
  {
    sub_C170B0(v4 + 200, v29);
    v29 = *((unsigned int *)a2 + 48);
  }
  v30 = (_QWORD *)a2[23];
  for ( i = &v30[v29]; i != v30; ++v30 )
  {
    a3(&v60, a4, *v30);
    v31 = *(unsigned int *)(v4 + 208);
    v32 = v31 + 1;
    v33 = *(_DWORD *)(v4 + 208);
    if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 212) )
    {
      v44 = *(_QWORD *)(v4 + 200);
      v52 = v4 + 200;
      if ( v44 > (unsigned __int64)&v60 || (unsigned __int64)&v60 >= v44 + 24 * v31 )
      {
        sub_C170B0(v52, v32);
        v31 = *(unsigned int *)(v4 + 208);
        v34 = *(_QWORD *)(v4 + 200);
        v35 = &v60;
        v33 = *(_DWORD *)(v4 + 208);
      }
      else
      {
        sub_C170B0(v52, v32);
        v34 = *(_QWORD *)(v4 + 200);
        v31 = *(unsigned int *)(v4 + 208);
        v35 = (__int64 *)((char *)&v60 + v34 - v44);
        v33 = *(_DWORD *)(v4 + 208);
      }
    }
    else
    {
      v34 = *(_QWORD *)(v4 + 200);
      v35 = &v60;
    }
    v36 = (_QWORD *)(v34 + 24 * v31);
    if ( v36 )
    {
      v37 = *v35;
      *v35 = 0;
      *v36 = v37;
      v38 = v35[1];
      v35[1] = 0;
      v36[1] = v38;
      v39 = v35[2];
      v35[2] = 0;
      v36[2] = v39;
      v33 = *(_DWORD *)(v4 + 208);
    }
    v40 = v61;
    v41 = v60;
    *(_DWORD *)(v4 + 208) = v33 + 1;
    if ( v40 != v41 )
    {
      do
      {
        v42 = *(_QWORD **)(v41 + 8);
        if ( v42 )
        {
          if ( (_QWORD *)*v42 != v42 + 2 )
            j_j___libc_free_0(*v42, v42[2] + 1LL);
          j_j___libc_free_0(v42, 32);
        }
        v41 += 32;
      }
      while ( v40 != v41 );
      v41 = v60;
    }
    if ( v41 )
      j_j___libc_free_0(v41, v62 - v41);
  }
  return v4;
}
