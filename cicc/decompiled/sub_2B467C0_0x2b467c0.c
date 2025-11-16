// Function: sub_2B467C0
// Address: 0x2b467c0
//
void __fastcall sub_2B467C0(unsigned int *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r13
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  const __m128i *v14; // r14
  const __m128i *v15; // r13
  const __m128i *v16; // rdx
  const __m128i *v17; // rax
  __int32 v18; // eax
  __int64 v19; // rax
  __int64 *v20; // r13
  __int64 *v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  _QWORD *v25; // r14
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  size_t v29; // rdx
  _QWORD *v30; // r13
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  __int64 *v34; // r13
  __int64 *v35; // r12
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  __int64 *v39; // r15
  _BYTE *v40; // r14
  __int64 v41; // rdx
  __int64 *v42; // r13
  __int64 v43; // rdx
  _BYTE *v44; // rsi
  __int64 *v45; // rdi
  _BYTE *v46; // rax
  __int64 *v47; // rdi
  size_t v48; // rdx
  _QWORD *v49; // r13
  unsigned __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned int v53; // r14d
  _BYTE *v54; // r13
  __int64 v55; // rax
  size_t *v56; // r14
  __int64 v57; // rdx
  _BYTE *v58; // rsi
  size_t *v59; // rdi
  _BYTE *v60; // rax
  size_t *v61; // rdi
  size_t v62; // rdx
  __int64 v63; // [rsp-60h] [rbp-60h]
  __int64 v64; // [rsp-58h] [rbp-58h]
  __int64 v65; // [rsp-58h] [rbp-58h]
  unsigned int v66; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v67; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 == a2 )
    return;
  v6 = (__int64 *)(a2 + 4);
  v9 = a1[2];
  v10 = *(_QWORD *)a1;
  if ( *(unsigned int **)a2 != a2 + 4 )
  {
    v25 = (_QWORD *)(v10 + 224 * v9);
    if ( v25 != (_QWORD *)v10 )
    {
      do
      {
        v25 -= 28;
        v26 = v25[23];
        if ( (_QWORD *)v26 != v25 + 25 )
          j_j___libc_free_0(v26);
        v27 = v25[19];
        if ( (_QWORD *)v27 != v25 + 21 )
          j_j___libc_free_0(v27);
        v28 = v25[1];
        if ( (_QWORD *)v28 != v25 + 3 )
          _libc_free(v28);
      }
      while ( v25 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
    if ( (unsigned int *)v10 != a1 + 4 )
      _libc_free(v10);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    a1[2] = a2[2];
    a1[3] = a2[3];
    *(_QWORD *)a2 = v6;
    *((_QWORD *)a2 + 1) = 0;
    return;
  }
  v11 = a2[2];
  v66 = a2[2];
  if ( v11 > v9 )
  {
    if ( v11 > a1[3] )
    {
      v49 = (_QWORD *)(v10 + 224 * v9);
      while ( v49 != (_QWORD *)v10 )
      {
        while ( 1 )
        {
          v49 -= 28;
          v50 = v49[23];
          if ( (_QWORD *)v50 != v49 + 25 )
            j_j___libc_free_0(v50);
          v51 = v49[19];
          if ( (_QWORD *)v51 != v49 + 21 )
            j_j___libc_free_0(v51);
          v52 = v49[1];
          if ( (_QWORD *)v52 == v49 + 3 )
            break;
          _libc_free(v52);
          if ( v49 == (_QWORD *)v10 )
            goto LABEL_92;
        }
      }
LABEL_92:
      a1[2] = 0;
      v10 = sub_C8D7D0((__int64)a1, (__int64)(a1 + 4), v11, 0xE0u, &v67, a6);
      sub_D39280(a1, (const __m128i *)v10);
      v53 = v67;
      if ( a1 + 4 != *(unsigned int **)a1 )
        _libc_free(*(_QWORD *)a1);
      *(_QWORD *)a1 = v10;
      a1[3] = v53;
      v6 = *(__int64 **)a2;
      v11 = a2[2];
      v12 = *(_QWORD *)a2;
LABEL_6:
      v13 = 7 * v11;
      v14 = (const __m128i *)(v12 + 200);
      v15 = (const __m128i *)&v6[4 * v13];
      if ( v15 != (const __m128i *)v12 )
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v19 = v14[-13].m128i_i64[1];
            *(_DWORD *)(v10 + 16) = 0;
            *(_DWORD *)(v10 + 20) = 8;
            *(_QWORD *)v10 = v19;
            *(_QWORD *)(v10 + 8) = v10 + 24;
            if ( v14[-12].m128i_i32[2] )
              sub_2B09D50(v10 + 8, (char **)&v14[-12], v12, a4, a5, a6);
            *(_QWORD *)(v10 + 152) = v10 + 168;
            v16 = (const __m128i *)v14[-3].m128i_i64[0];
            if ( &v14[-2] == v16 )
            {
              *(__m128i *)(v10 + 168) = _mm_loadu_si128(v14 - 2);
            }
            else
            {
              *(_QWORD *)(v10 + 152) = v16;
              *(_QWORD *)(v10 + 168) = v14[-2].m128i_i64[0];
            }
            v12 = v14[-3].m128i_i64[1];
            *(_QWORD *)(v10 + 160) = v12;
            v14[-3].m128i_i64[0] = (__int64)v14[-2].m128i_i64;
            v14[-3].m128i_i64[1] = 0;
            v14[-2].m128i_i8[0] = 0;
            *(_QWORD *)(v10 + 184) = v10 + 200;
            v17 = (const __m128i *)v14[-1].m128i_i64[0];
            if ( v17 == v14 )
            {
              *(__m128i *)(v10 + 200) = _mm_loadu_si128(v14);
            }
            else
            {
              *(_QWORD *)(v10 + 184) = v17;
              *(_QWORD *)(v10 + 200) = v14->m128i_i64[0];
            }
            *(_QWORD *)(v10 + 192) = v14[-1].m128i_i64[1];
            v18 = v14[1].m128i_i32[0];
            v14[-1].m128i_i64[0] = (__int64)v14;
            v14[-1].m128i_i64[1] = 0;
            v14->m128i_i8[0] = 0;
            *(_DWORD *)(v10 + 216) = v18;
          }
          v10 += 224LL;
          if ( v15 == (const __m128i *)&v14[1].m128i_u64[1] )
            break;
          v14 += 14;
        }
      }
      a1[2] = v66;
      v20 = *(__int64 **)a2;
      v21 = (__int64 *)(*(_QWORD *)a2 + 224LL * a2[2]);
      if ( *(__int64 **)a2 != v21 )
      {
        do
        {
          v21 -= 28;
          v22 = v21[23];
          if ( (__int64 *)v22 != v21 + 25 )
            j_j___libc_free_0(v22);
          v23 = v21[19];
          if ( (__int64 *)v23 != v21 + 21 )
            j_j___libc_free_0(v23);
          v24 = v21[1];
          if ( (__int64 *)v24 != v21 + 3 )
            _libc_free(v24);
        }
        while ( v20 != v21 );
      }
      goto LABEL_28;
    }
    v12 = (__int64)(a2 + 4);
    if ( !a1[2] )
      goto LABEL_6;
    v39 = (__int64 *)(v10 + 200);
    v40 = a2 + 46;
    v41 = 7 * v9;
    v64 = 224 * v9;
    v42 = &v39[28 * v9];
    while ( 1 )
    {
      *((_DWORD *)v39 - 50) = *((_DWORD *)v40 - 42);
      *((_BYTE *)v39 - 196) = *(v40 - 164);
      sub_2B09D50((__int64)(v39 - 24), (char **)v40 - 20, v41, a4, a5, a6);
      v46 = (_BYTE *)*((_QWORD *)v40 - 2);
      v47 = (__int64 *)*(v39 - 6);
      if ( v46 == v40 )
      {
        v48 = *((_QWORD *)v40 - 1);
        if ( v48 )
        {
          if ( v48 == 1 )
            *(_BYTE *)v47 = *v40;
          else
            memcpy(v47, v40, v48);
          v48 = *((_QWORD *)v40 - 1);
          v47 = (__int64 *)*(v39 - 6);
        }
        *(v39 - 5) = v48;
        *((_BYTE *)v47 + v48) = 0;
        v47 = (__int64 *)*((_QWORD *)v40 - 2);
      }
      else
      {
        if ( v47 == v39 - 4 )
        {
          *(v39 - 6) = (__int64)v46;
          *(v39 - 5) = *((_QWORD *)v40 - 1);
          *(v39 - 4) = *(_QWORD *)v40;
        }
        else
        {
          *(v39 - 6) = (__int64)v46;
          v43 = *(v39 - 4);
          *(v39 - 5) = *((_QWORD *)v40 - 1);
          *(v39 - 4) = *(_QWORD *)v40;
          if ( v47 )
          {
            *((_QWORD *)v40 - 2) = v47;
            *(_QWORD *)v40 = v43;
            goto LABEL_63;
          }
        }
        *((_QWORD *)v40 - 2) = v40;
        v47 = (__int64 *)v40;
      }
LABEL_63:
      *((_QWORD *)v40 - 1) = 0;
      *(_BYTE *)v47 = 0;
      v44 = (_BYTE *)*((_QWORD *)v40 + 2);
      v45 = (__int64 *)*(v39 - 2);
      if ( v44 == v40 + 32 )
      {
        v41 = *((_QWORD *)v40 + 3);
        if ( v41 )
        {
          if ( v41 == 1 )
            *(_BYTE *)v45 = v40[32];
          else
            memcpy(v45, v44, v41);
          v41 = *((_QWORD *)v40 + 3);
          v45 = (__int64 *)*(v39 - 2);
        }
        *(v39 - 1) = v41;
        *((_BYTE *)v45 + v41) = 0;
        v45 = (__int64 *)*((_QWORD *)v40 + 2);
        goto LABEL_67;
      }
      if ( v45 == v39 )
      {
        *(v39 - 2) = (__int64)v44;
        *(v39 - 1) = *((_QWORD *)v40 + 3);
        v41 = *((_QWORD *)v40 + 4);
        *v39 = v41;
LABEL_81:
        *((_QWORD *)v40 + 2) = v40 + 32;
        v45 = (__int64 *)(v40 + 32);
        goto LABEL_67;
      }
      *(v39 - 2) = (__int64)v44;
      v41 = *v39;
      *(v39 - 1) = *((_QWORD *)v40 + 3);
      a4 = *((_QWORD *)v40 + 4);
      *v39 = a4;
      if ( !v45 )
        goto LABEL_81;
      *((_QWORD *)v40 + 2) = v45;
      *((_QWORD *)v40 + 4) = v41;
LABEL_67:
      *((_QWORD *)v40 + 3) = 0;
      v39 += 28;
      v40 += 224;
      *(_BYTE *)v45 = 0;
      *((_DWORD *)v39 - 52) = *((_DWORD *)v40 - 44);
      if ( v39 == v42 )
      {
        v6 = *(__int64 **)a2;
        v11 = a2[2];
        v10 = v64 + *(_QWORD *)a1;
        v12 = *(_QWORD *)a2 + v64;
        goto LABEL_6;
      }
    }
  }
  v29 = *(_QWORD *)a1;
  if ( !a2[2] )
    goto LABEL_42;
  v54 = a2 + 46;
  v55 = 7 * v11;
  v56 = (size_t *)(v10 + 200);
  v63 = 32 * v55;
  v65 = v10 + 200 + 32 * v55;
  do
  {
    *((_DWORD *)v56 - 50) = *((_DWORD *)v54 - 42);
    *((_BYTE *)v56 - 196) = *(v54 - 164);
    sub_2B09D50((__int64)(v56 - 24), (char **)v54 - 20, v29, a4, a5, a6);
    v60 = (_BYTE *)*((_QWORD *)v54 - 2);
    v61 = (size_t *)*(v56 - 6);
    if ( v60 == v54 )
    {
      v62 = *((_QWORD *)v54 - 1);
      if ( v62 )
      {
        if ( v62 == 1 )
          *(_BYTE *)v61 = *v54;
        else
          memcpy(v61, v54, v62);
        v62 = *((_QWORD *)v54 - 1);
        v61 = (size_t *)*(v56 - 6);
      }
      *(v56 - 5) = v62;
      *((_BYTE *)v61 + v62) = 0;
      v61 = (size_t *)*((_QWORD *)v54 - 2);
    }
    else
    {
      if ( v61 == v56 - 4 )
      {
        *(v56 - 6) = (size_t)v60;
        *(v56 - 5) = *((_QWORD *)v54 - 1);
        *(v56 - 4) = *(_QWORD *)v54;
      }
      else
      {
        *(v56 - 6) = (size_t)v60;
        v57 = *(v56 - 4);
        *(v56 - 5) = *((_QWORD *)v54 - 1);
        *(v56 - 4) = *(_QWORD *)v54;
        if ( v61 )
        {
          *((_QWORD *)v54 - 2) = v61;
          *(_QWORD *)v54 = v57;
          goto LABEL_100;
        }
      }
      *((_QWORD *)v54 - 2) = v54;
      v61 = (size_t *)v54;
    }
LABEL_100:
    *((_QWORD *)v54 - 1) = 0;
    *(_BYTE *)v61 = 0;
    v58 = (_BYTE *)*((_QWORD *)v54 + 2);
    v59 = (size_t *)*(v56 - 2);
    if ( v58 == v54 + 32 )
    {
      v29 = *((_QWORD *)v54 + 3);
      if ( v29 )
      {
        if ( v29 == 1 )
          *(_BYTE *)v59 = v54[32];
        else
          memcpy(v59, v58, v29);
        v29 = *((_QWORD *)v54 + 3);
        v59 = (size_t *)*(v56 - 2);
      }
      *(v56 - 1) = v29;
      *((_BYTE *)v59 + v29) = 0;
      v59 = (size_t *)*((_QWORD *)v54 + 2);
    }
    else
    {
      if ( v59 == v56 )
      {
        *(v56 - 2) = (size_t)v58;
        *(v56 - 1) = *((_QWORD *)v54 + 3);
        v29 = *((_QWORD *)v54 + 4);
        *v56 = v29;
      }
      else
      {
        *(v56 - 2) = (size_t)v58;
        v29 = *v56;
        *(v56 - 1) = *((_QWORD *)v54 + 3);
        *v56 = *((_QWORD *)v54 + 4);
        if ( v59 )
        {
          *((_QWORD *)v54 + 2) = v59;
          *((_QWORD *)v54 + 4) = v29;
          goto LABEL_104;
        }
      }
      *((_QWORD *)v54 + 2) = v54 + 32;
      v59 = (size_t *)(v54 + 32);
    }
LABEL_104:
    *((_QWORD *)v54 + 3) = 0;
    v56 += 28;
    v54 += 224;
    *(_BYTE *)v59 = 0;
    *((_DWORD *)v56 - 52) = *((_DWORD *)v54 - 44);
  }
  while ( v56 != (size_t *)v65 );
  v29 = *(_QWORD *)a1;
  v9 = a1[2];
  v10 += v63;
LABEL_42:
  v30 = (_QWORD *)(v29 + 224 * v9);
  while ( (_QWORD *)v10 != v30 )
  {
    v30 -= 28;
    v31 = v30[23];
    if ( (_QWORD *)v31 != v30 + 25 )
      j_j___libc_free_0(v31);
    v32 = v30[19];
    if ( (_QWORD *)v32 != v30 + 21 )
      j_j___libc_free_0(v32);
    v33 = v30[1];
    if ( (_QWORD *)v33 != v30 + 3 )
      _libc_free(v33);
  }
  a1[2] = v66;
  v34 = *(__int64 **)a2;
  v35 = (__int64 *)(*(_QWORD *)a2 + 224LL * a2[2]);
  if ( *(__int64 **)a2 != v35 )
  {
    do
    {
      v35 -= 28;
      v36 = v35[23];
      if ( (__int64 *)v36 != v35 + 25 )
        j_j___libc_free_0(v36);
      v37 = v35[19];
      if ( (__int64 *)v37 != v35 + 21 )
        j_j___libc_free_0(v37);
      v38 = v35[1];
      if ( (__int64 *)v38 != v35 + 3 )
        _libc_free(v38);
    }
    while ( v34 != v35 );
  }
LABEL_28:
  a2[2] = 0;
}
