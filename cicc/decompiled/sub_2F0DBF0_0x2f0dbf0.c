// Function: sub_2F0DBF0
// Address: 0x2f0dbf0
//
void __fastcall sub_2F0DBF0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v4; // r15
  __int64 v5; // r12
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 *v9; // r14
  __int64 *v10; // rbx
  __int64 v11; // rdx
  _BYTE *v12; // rsi
  _BYTE *v13; // rsi
  __m128i v14; // xmm3
  __int64 v15; // rdx
  __int64 v16; // rdx
  __m128i v17; // xmm4
  __int64 v18; // r12
  _BYTE *v19; // rdi
  __int64 *v20; // rax
  size_t v21; // rdx
  __int64 v22; // rbx
  __m128i *v23; // rsi
  size_t v24; // rdx
  __m128i v25; // xmm6
  int v26; // eax
  __int64 v27; // r12
  __int64 v28; // rdx
  _BYTE *v29; // rsi
  __int64 v30; // rdx
  __m128i v31; // xmm0
  __int64 v32; // rax
  __m128i v33; // xmm1
  bool v34; // r13
  __int64 *v35; // rbx
  __m128i *v36; // rdi
  __int64 *v37; // rax
  size_t v38; // rdx
  __m128i *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r8
  int v42; // eax
  __int64 *v43; // r12
  __int64 v47; // [rsp+20h] [rbp-140h]
  _BYTE *v49; // [rsp+30h] [rbp-130h]
  bool v50; // [rsp+40h] [rbp-120h]
  __m128i *v51; // [rsp+40h] [rbp-120h]
  __int64 v52; // [rsp+70h] [rbp-F0h]
  __m128i *v53; // [rsp+78h] [rbp-E8h]
  size_t n; // [rsp+80h] [rbp-E0h]
  __m128i v55; // [rsp+88h] [rbp-D8h] BYREF
  __m128i v56; // [rsp+98h] [rbp-C8h] BYREF
  int v57; // [rsp+A8h] [rbp-B8h]
  __int64 v58; // [rsp+B0h] [rbp-B0h]
  __int64 v59[2]; // [rsp+B8h] [rbp-A8h] BYREF
  _QWORD v60[2]; // [rsp+C8h] [rbp-98h] BYREF
  __m128i v61; // [rsp+D8h] [rbp-88h]
  int v62; // [rsp+E8h] [rbp-78h]
  __int64 v63; // [rsp+F0h] [rbp-70h]
  __int64 v64[2]; // [rsp+F8h] [rbp-68h] BYREF
  _QWORD v65[2]; // [rsp+108h] [rbp-58h] BYREF
  __m128i v66; // [rsp+118h] [rbp-48h]
  int v67; // [rsp+128h] [rbp-38h]

  v4 = (__int64 *)(a1 + (a2 << 6));
  v47 = (a3 - 1) / 2;
  if ( a2 >= v47 )
  {
    v22 = a2;
    goto LABEL_25;
  }
  v5 = a2;
  v49 = v4 + 3;
  while ( 1 )
  {
    v8 = 2 * (v5 + 1);
    v9 = (__int64 *)(a1 + ((v5 + 1) << 7));
    v10 = (__int64 *)(a1 + ((v8 - 1) << 6));
    v11 = *v10;
    v12 = (_BYTE *)v10[1];
    v64[0] = (__int64)v65;
    v63 = v11;
    sub_2F07250(v64, v12, (__int64)&v12[v10[2]]);
    v13 = (_BYTE *)v9[1];
    v14 = _mm_loadu_si128((const __m128i *)(v10 + 5));
    v67 = *((_DWORD *)v10 + 14);
    v15 = *v9;
    v59[0] = (__int64)v60;
    v58 = v15;
    v16 = v9[2];
    v66 = v14;
    sub_2F07250(v59, v13, (__int64)&v13[v16]);
    v17 = _mm_loadu_si128((const __m128i *)(v9 + 5));
    v62 = *((_DWORD *)v9 + 14);
    v61 = v17;
    v50 = (unsigned int)v58 < (unsigned int)v63;
    if ( (_DWORD)v58 == (_DWORD)v63 )
      v50 = HIDWORD(v58) < HIDWORD(v63);
    if ( (_QWORD *)v59[0] != v60 )
      j_j___libc_free_0(v59[0]);
    if ( (_QWORD *)v64[0] != v65 )
      j_j___libc_free_0(v64[0]);
    if ( v50 )
    {
      --v8;
      v9 = v10;
    }
    v18 = a1 + (v5 << 6);
    v19 = *(_BYTE **)(v18 + 8);
    *(_QWORD *)v18 = *v9;
    v20 = (__int64 *)v9[1];
    if ( v20 == v9 + 3 )
    {
      v21 = v9[2];
      if ( v21 )
      {
        if ( v21 == 1 )
          *v19 = *((_BYTE *)v9 + 24);
        else
          memcpy(v19, v9 + 3, v21);
        v21 = v9[2];
        v19 = *(_BYTE **)(v18 + 8);
      }
      *(_QWORD *)(v18 + 16) = v21;
      v19[v21] = 0;
      v19 = (_BYTE *)v9[1];
    }
    else
    {
      if ( v19 == v49 )
      {
        *(_QWORD *)(v18 + 8) = v20;
        *(_QWORD *)(v18 + 16) = v9[2];
        *(_QWORD *)(v18 + 24) = v9[3];
      }
      else
      {
        *(_QWORD *)(v18 + 8) = v20;
        v7 = *(_QWORD *)(v18 + 24);
        *(_QWORD *)(v18 + 16) = v9[2];
        *(_QWORD *)(v18 + 24) = v9[3];
        if ( v19 )
        {
          v9[1] = (__int64)v19;
          v9[3] = v7;
          goto LABEL_6;
        }
      }
      v9[1] = (__int64)(v9 + 3);
      v19 = v9 + 3;
    }
LABEL_6:
    v9[2] = 0;
    *v19 = 0;
    *(__m128i *)(v18 + 40) = _mm_loadu_si128((const __m128i *)(v9 + 5));
    *(_DWORD *)(v18 + 56) = *((_DWORD *)v9 + 14);
    if ( v8 >= v47 )
      break;
    v49 = v9 + 3;
    v5 = v8;
  }
  v22 = v8;
  v4 = v9;
LABEL_25:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v22 )
  {
    v22 = 2 * v22 + 1;
    v43 = (__int64 *)(a1 + (v22 << 6));
    *v4 = *v43;
    sub_2F074A0((__int64)(v4 + 1), (__int64)(v43 + 1));
    *(__m128i *)(v4 + 5) = _mm_loadu_si128((const __m128i *)(v43 + 5));
    *((_DWORD *)v4 + 14) = *((_DWORD *)v43 + 14);
    v4 = v43;
  }
  v23 = (__m128i *)a4[1];
  v52 = *a4;
  v53 = &v55;
  if ( v23 == (__m128i *)(a4 + 3) )
  {
    v23 = &v55;
    v55 = _mm_loadu_si128((const __m128i *)(a4 + 3));
  }
  else
  {
    v53 = (__m128i *)a4[1];
    v55.m128i_i64[0] = a4[3];
  }
  v24 = a4[2];
  v25 = _mm_loadu_si128((const __m128i *)(a4 + 5));
  a4[1] = (__int64)(a4 + 3);
  v26 = *((_DWORD *)a4 + 14);
  a4[2] = 0;
  n = v24;
  v57 = v26;
  *((_BYTE *)a4 + 24) = 0;
  v56 = v25;
  v51 = (__m128i *)(v4 + 3);
  v27 = (v22 - 1) / 2;
  if ( v22 > a2 )
  {
    while ( 2 )
    {
      v58 = v52;
      v4 = (__int64 *)(a1 + (v27 << 6));
      v59[0] = (__int64)v60;
      sub_2F07250(v59, v23, (__int64)v23->m128i_i64 + v24);
      v29 = (_BYTE *)v4[1];
      v30 = v4[2];
      v31 = _mm_loadu_si128(&v56);
      v62 = v57;
      v32 = *v4;
      v61 = v31;
      v63 = v32;
      v64[0] = (__int64)v65;
      sub_2F07250(v64, v29, (__int64)&v29[v30]);
      v33 = _mm_loadu_si128((const __m128i *)(v4 + 5));
      v67 = *((_DWORD *)v4 + 14);
      v66 = v33;
      v34 = (unsigned int)v63 < (unsigned int)v58;
      if ( (_DWORD)v63 == (_DWORD)v58 )
        v34 = HIDWORD(v63) < HIDWORD(v58);
      if ( (_QWORD *)v64[0] != v65 )
        j_j___libc_free_0(v64[0]);
      if ( (_QWORD *)v59[0] != v60 )
        j_j___libc_free_0(v59[0]);
      v35 = (__int64 *)(a1 + (v22 << 6));
      v36 = (__m128i *)v35[1];
      if ( !v34 )
      {
        v24 = n;
        v23 = v53;
        v4 = v35;
        goto LABEL_53;
      }
      *v35 = *v4;
      v37 = (__int64 *)v4[1];
      if ( v37 == v4 + 3 )
      {
        v38 = v4[2];
        if ( v38 )
        {
          if ( v38 == 1 )
            v36->m128i_i8[0] = *((_BYTE *)v4 + 24);
          else
            memcpy(v36, v4 + 3, v38);
          v38 = v4[2];
          v36 = (__m128i *)v35[1];
        }
        v35[2] = v38;
        v36->m128i_i8[v38] = 0;
        v36 = (__m128i *)v4[1];
        goto LABEL_35;
      }
      if ( v51 == v36 )
      {
        v35[1] = (__int64)v37;
        v35[2] = v4[2];
        v35[3] = v4[3];
      }
      else
      {
        v35[1] = (__int64)v37;
        v28 = v35[3];
        v35[2] = v4[2];
        v35[3] = v4[3];
        if ( v36 )
        {
          v4[1] = (__int64)v36;
          v4[3] = v28;
LABEL_35:
          v4[2] = 0;
          v36->m128i_i8[0] = 0;
          *(__m128i *)(v35 + 5) = _mm_loadu_si128((const __m128i *)(v4 + 5));
          *((_DWORD *)v35 + 14) = *((_DWORD *)v4 + 14);
          if ( a2 >= v27 )
          {
            v51 = (__m128i *)(v4 + 3);
            v24 = n;
            v23 = v53;
            goto LABEL_53;
          }
          v51 = (__m128i *)(v4 + 3);
          v23 = v53;
          v22 = v27;
          v27 = (v27 - 1) / 2;
          v24 = n;
          continue;
        }
      }
      break;
    }
    v4[1] = (__int64)(v4 + 3);
    v36 = (__m128i *)(v4 + 3);
    goto LABEL_35;
  }
LABEL_53:
  v39 = (__m128i *)v4[1];
  *v4 = v52;
  if ( v23 == &v55 )
  {
    if ( v24 )
    {
      if ( v24 == 1 )
        v39->m128i_i8[0] = v55.m128i_i8[0];
      else
        memcpy(v39, &v55, v24);
      v24 = n;
      v39 = (__m128i *)v4[1];
    }
    v4[2] = v24;
    v39->m128i_i8[v24] = 0;
    v39 = v53;
  }
  else
  {
    v40 = v55.m128i_i64[0];
    if ( v39 == v51 )
    {
      v4[1] = (__int64)v23;
      v4[2] = v24;
      v4[3] = v40;
    }
    else
    {
      v41 = v4[3];
      v4[1] = (__int64)v23;
      v4[2] = v24;
      v4[3] = v40;
      if ( v39 )
      {
        v53 = v39;
        v55.m128i_i64[0] = v41;
        goto LABEL_57;
      }
    }
    v53 = &v55;
    v39 = &v55;
  }
LABEL_57:
  v39->m128i_i8[0] = 0;
  v42 = v57;
  *(__m128i *)(v4 + 5) = _mm_loadu_si128(&v56);
  *((_DWORD *)v4 + 14) = v42;
  if ( v53 != &v55 )
    j_j___libc_free_0((unsigned __int64)v53);
}
