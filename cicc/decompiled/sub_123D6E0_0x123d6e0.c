// Function: sub_123D6E0
// Address: 0x123d6e0
//
__int64 __fastcall sub_123D6E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  __m128i *v13; // r8
  bool v14; // zf
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  void *v17; // rcx
  void *v18; // r10
  size_t v19; // rdx
  void *v20; // rax
  __int64 v21; // r14
  __int64 *v22; // r15
  const __m128i *v23; // r11
  unsigned int v24; // eax
  const __m128i *v25; // rcx
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __m128i *v28; // rdx
  const __m128i *v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rax
  const __m128i *v34; // rbx
  const __m128i **v35; // r14
  __int64 *v36; // rcx
  __int64 v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int8 v41; // [rsp+17h] [rbp-E9h]
  __int64 v42; // [rsp+20h] [rbp-E0h]
  __m128i *v43; // [rsp+28h] [rbp-D8h]
  __int64 v44; // [rsp+28h] [rbp-D8h]
  __m128i *v45; // [rsp+30h] [rbp-D0h]
  size_t v46; // [rsp+30h] [rbp-D0h]
  __int64 v47; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v48; // [rsp+38h] [rbp-C8h]
  void *v49; // [rsp+38h] [rbp-C8h]
  __int64 *v50; // [rsp+38h] [rbp-C8h]
  __int64 v51; // [rsp+48h] [rbp-B8h] BYREF
  unsigned int v52; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-A8h]
  __int64 v54; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v55; // [rsp+68h] [rbp-98h]
  __m128i v56; // [rsp+70h] [rbp-90h] BYREF
  void *src; // [rsp+80h] [rbp-80h]
  _BYTE *v58; // [rsp+88h] [rbp-78h]
  __int64 v59; // [rsp+90h] [rbp-70h]
  __int64 v60; // [rsp+A0h] [rbp-60h] BYREF
  int v61; // [rsp+A8h] [rbp-58h] BYREF
  _QWORD *v62; // [rsp+B0h] [rbp-50h]
  int *v63; // [rsp+B8h] [rbp-48h]
  int *v64; // [rsp+C0h] [rbp-40h]
  __int64 v65; // [rsp+C8h] [rbp-38h]

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v61 = 0;
    v63 = &v61;
    v64 = &v61;
    v62 = 0;
    v65 = 0;
    v6 = a3;
    v7 = a1 + 176;
    v8 = v6;
    while ( 1 )
    {
      v9 = a1;
      v10 = (__int64)(*(_QWORD *)(v8 + 8) - *(_QWORD *)v8) >> 3;
      v11 = (__int64)&v56;
      src = 0;
      v58 = 0;
      v59 = 0;
      v12 = sub_1239370(a1, v56.m128i_i64, &v60, -858993459 * (int)v10);
      if ( v12 )
      {
        v41 = v12;
        if ( src )
          j_j___libc_free_0(src, v59 - (_QWORD)src);
        goto LABEL_49;
      }
      v13 = *(__m128i **)(v8 + 8);
      if ( v13 == *(__m128i **)(v8 + 16) )
      {
        sub_D78F50(v8, *(const __m128i **)(v8 + 8), &v56);
        v18 = src;
      }
      else
      {
        if ( v13 )
        {
          *v13 = _mm_loadu_si128(&v56);
          v15 = v58 - (_BYTE *)src;
          v14 = v58 == src;
          v13[1].m128i_i64[0] = 0;
          v13[1].m128i_i64[1] = 0;
          v13[2].m128i_i64[0] = 0;
          if ( v14 )
          {
            v17 = 0;
          }
          else
          {
            if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_59:
              sub_4261EA(v9, v11, v15);
            v45 = v13;
            v48 = v15;
            v16 = sub_22077B0(v15);
            v15 = v48;
            v13 = v45;
            v17 = (void *)v16;
          }
          v13[1].m128i_i64[0] = (__int64)v17;
          v13[2].m128i_i64[0] = (__int64)v17 + v15;
          v13[1].m128i_i64[1] = (__int64)v17;
          v18 = src;
          v19 = v58 - (_BYTE *)src;
          if ( v58 != src )
          {
            v43 = v13;
            v46 = v58 - (_BYTE *)src;
            v49 = src;
            v20 = memmove(v17, src, v19);
            v13 = v43;
            v19 = v46;
            v18 = v49;
            v17 = v20;
          }
          v13[1].m128i_i64[1] = (__int64)v17 + v19;
          v13 = *(__m128i **)(v8 + 8);
        }
        else
        {
          v18 = src;
        }
        *(_QWORD *)(v8 + 8) = (char *)v13 + 40;
      }
      if ( v18 )
        j_j___libc_free_0(v18, v59 - (_QWORD)v18);
      if ( *(_DWORD *)(a1 + 240) != 4 )
        break;
      *(_DWORD *)(a1 + 240) = sub_1205200(v7);
    }
    v11 = 13;
    v9 = a1;
    v41 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( !v41 )
    {
      v21 = (__int64)v63;
      v47 = a1 + 1656;
      if ( v63 != &v61 )
      {
        v42 = a1;
        v22 = (__int64 *)v8;
        do
        {
          v23 = *(const __m128i **)(v21 + 48);
          v24 = *(_DWORD *)(v21 + 32);
          v53 = 0;
          v54 = 0;
          v25 = *(const __m128i **)(v21 + 40);
          v52 = v24;
          v55 = 0;
          v26 = (char *)v23 - (char *)v25;
          if ( v23 == v25 )
          {
            v27 = 0;
          }
          else
          {
            if ( v26 > 0x7FFFFFFFFFFFFFF0LL )
              goto LABEL_59;
            v27 = sub_22077B0((char *)v23 - (char *)v25);
            v23 = *(const __m128i **)(v21 + 48);
            v25 = *(const __m128i **)(v21 + 40);
          }
          v53 = v27;
          v54 = v27;
          v55 = v27 + v26;
          if ( v23 == v25 )
          {
            v29 = (const __m128i *)v27;
          }
          else
          {
            v28 = (__m128i *)v27;
            v29 = (const __m128i *)(v27 + (char *)v23 - (char *)v25);
            do
            {
              if ( v28 )
                *v28 = _mm_loadu_si128(v25);
              ++v28;
              ++v25;
            }
            while ( v29 != v28 );
          }
          v54 = (__int64)v29;
          v30 = *(_QWORD *)(v42 + 1664);
          if ( v30 )
          {
            v31 = v47;
            do
            {
              while ( 1 )
              {
                v11 = *(_QWORD *)(v30 + 16);
                v32 = *(_QWORD *)(v30 + 24);
                if ( *(_DWORD *)(v30 + 32) >= v52 )
                  break;
                v30 = *(_QWORD *)(v30 + 24);
                if ( !v32 )
                  goto LABEL_35;
              }
              v31 = v30;
              v30 = *(_QWORD *)(v30 + 16);
            }
            while ( v11 );
LABEL_35:
            if ( v31 != v47 && v52 >= *(_DWORD *)(v31 + 32) )
              goto LABEL_38;
          }
          else
          {
            v31 = v47;
          }
          v11 = v31;
          v56.m128i_i64[0] = (__int64)&v52;
          v33 = sub_123CD60((_QWORD *)(v42 + 1648), v31, (unsigned int **)&v56);
          v29 = (const __m128i *)v54;
          v31 = v33;
          v27 = v53;
LABEL_38:
          if ( (const __m128i *)v27 != v29 )
          {
            v44 = v21;
            v34 = (const __m128i *)v27;
            v35 = (const __m128i **)(v31 + 40);
            v36 = &v56.m128i_i64[1];
            v37 = v31;
            do
            {
              while ( 1 )
              {
                v38 = 5LL * v34->m128i_u32[0];
                v39 = *v22;
                v56 = _mm_loadu_si128(v34);
                v11 = *(_QWORD *)(v37 + 48);
                v40 = v39 + 8 * v38;
                v51 = v40;
                if ( v11 != *(_QWORD *)(v37 + 56) )
                  break;
                ++v34;
                v50 = v36;
                sub_12149F0(v35, (const __m128i *)v11, &v51, v36);
                v36 = v50;
                if ( v29 == v34 )
                  goto LABEL_45;
              }
              if ( v11 )
              {
                *(_QWORD *)v11 = v40;
                *(_QWORD *)(v11 + 8) = v56.m128i_i64[1];
                v11 = *(_QWORD *)(v37 + 48);
              }
              v11 += 16;
              ++v34;
              *(_QWORD *)(v37 + 48) = v11;
            }
            while ( v29 != v34 );
LABEL_45:
            v21 = v44;
            v29 = (const __m128i *)v53;
          }
          if ( v29 )
          {
            v11 = v55 - (_QWORD)v29;
            j_j___libc_free_0(v29, v55 - (_QWORD)v29);
          }
          v9 = v21;
          v21 = sub_220EEE0(v21);
        }
        while ( (int *)v21 != &v61 );
      }
    }
LABEL_49:
    sub_1207E40(v62);
  }
  return v41;
}
