// Function: sub_FFCE90
// Address: 0xffce90
//
__int64 __fastcall sub_FFCE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  bool v7; // zf
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // rax
  __m128i *v18; // rdx
  const __m128i *v19; // r12
  unsigned __int64 v20; // r10
  unsigned __int64 v21; // r12
  unsigned __int64 *v22; // rsi
  char *v23; // r14
  char *v24; // r15
  char *v25; // rdi
  char *v26; // rdi
  char *v27; // r15
  char *v28; // r14
  char *v29; // rdi
  char *v30; // rdi
  __int64 v31; // [rsp+8h] [rbp-528h]
  __m128i v32; // [rsp+10h] [rbp-520h] BYREF
  unsigned __int64 v33; // [rsp+20h] [rbp-510h]
  __int64 v34; // [rsp+28h] [rbp-508h]
  unsigned __int64 *v35; // [rsp+30h] [rbp-500h] BYREF
  __int64 v36; // [rsp+38h] [rbp-4F8h]
  _BYTE v37[512]; // [rsp+40h] [rbp-4F0h] BYREF
  unsigned __int64 *v38; // [rsp+240h] [rbp-2F0h] BYREF
  __int64 v39; // [rsp+248h] [rbp-2E8h]
  char *v40; // [rsp+250h] [rbp-2E0h] BYREF
  unsigned int v41; // [rsp+258h] [rbp-2D8h]
  char v42; // [rsp+370h] [rbp-1C0h] BYREF
  char v43; // [rsp+378h] [rbp-1B8h]
  char *v44; // [rsp+380h] [rbp-1B0h] BYREF
  unsigned int v45; // [rsp+388h] [rbp-1A8h]
  char v46; // [rsp+4A0h] [rbp-90h] BYREF
  char *v47; // [rsp+4A8h] [rbp-88h]
  char v48; // [rsp+4B8h] [rbp-78h] BYREF

  result = *(_QWORD *)(a1 + 544);
  v7 = *(_BYTE *)(a1 + 560) == 1;
  v34 = result;
  if ( v7 && result )
  {
    while ( 1 )
    {
      v9 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)(a1 + 528);
      if ( result == v9 )
        return result;
      v10 = *(_QWORD *)a1 + 32 * result;
      v11 = *(_QWORD *)a1 + 32 * v9;
      if ( !*(_BYTE *)v10 )
        break;
      v38 = (unsigned __int64 *)&v40;
      v39 = 0x200000000LL;
      if ( v10 == v11 )
      {
        v14 = 0;
        v15 = (unsigned __int64)&v40;
      }
      else
      {
        v17 = v10 + 8;
        v15 = (unsigned __int64)&v40;
        v14 = 0;
        while ( 1 )
        {
          v19 = (const __m128i *)v17;
          if ( !*(_BYTE *)(v17 - 8) )
            break;
          v20 = v14 + 1;
          if ( v14 + 1 > (unsigned __int64)HIDWORD(v39) )
          {
            if ( v15 > v17 || v15 + 24 * v14 <= v17 )
            {
              v33 = v17;
              v32.m128i_i64[0] = v11;
              sub_C8D5F0((__int64)&v38, &v40, v20, 0x18u, v11, a6);
              v15 = (unsigned __int64)v38;
              v14 = (unsigned int)v39;
              v11 = v32.m128i_i64[0];
              v17 = v33;
            }
            else
            {
              v21 = v17 - v15;
              v33 = v11;
              v32.m128i_i64[0] = v17;
              sub_C8D5F0((__int64)&v38, &v40, v20, 0x18u, v11, a6);
              v15 = (unsigned __int64)v38;
              v14 = (unsigned int)v39;
              v11 = v33;
              v17 = v32.m128i_i64[0];
              v19 = (const __m128i *)((char *)v38 + v21);
            }
          }
          v18 = (__m128i *)(v15 + 24 * v14);
          *v18 = _mm_loadu_si128(v19);
          v18[1].m128i_i64[0] = v19[1].m128i_i64[0];
          a4 = v17 + 32;
          v15 = (unsigned __int64)v38;
          v14 = (unsigned int)(v39 + 1);
          LODWORD(v39) = v39 + 1;
          if ( v11 == v17 + 24 )
            break;
          v17 += 32LL;
        }
      }
      sub_FFC290(a1, (__int64 *)v15, v14, a4, v11, a6);
      v16 = v38;
      result = (unsigned int)v39;
      *(_QWORD *)(a1 + 528) += (unsigned int)v39;
      if ( v16 == (unsigned __int64 *)&v40 )
        goto LABEL_14;
LABEL_13:
      result = _libc_free(v16, v15);
LABEL_14:
      if ( !*(_QWORD *)(a1 + 544) )
        return result;
    }
    v12 = 0;
    v35 = (unsigned __int64 *)v37;
    v36 = 0x2000000000LL;
    if ( v10 == v11 )
    {
      v22 = (unsigned __int64 *)v37;
    }
    else
    {
      do
      {
        if ( *(_BYTE *)v10 )
          break;
        v13 = _mm_loadu_si128((const __m128i *)(v10 + 8));
        if ( v12 + 1 > (unsigned __int64)HIDWORD(v36) )
        {
          v31 = v11;
          v33 = v10;
          v32 = v13;
          sub_C8D5F0((__int64)&v35, v37, v12 + 1, 0x10u, v11, v12 + 1);
          v12 = (unsigned int)v36;
          v11 = v31;
          v10 = v33;
          v13 = _mm_load_si128(&v32);
        }
        v10 += 32;
        *(__m128i *)&v35[2 * v12] = v13;
        v12 = (unsigned int)(v36 + 1);
        LODWORD(v36) = v36 + 1;
      }
      while ( v10 != v11 );
      v22 = v35;
    }
    sub_B26290((__int64)&v38, v22, v12, 1u);
    v15 = (unsigned __int64)&v38;
    sub_B24D40(v34, (__int64)&v38, 0);
    if ( v47 != &v48 )
      _libc_free(v47, &v38);
    if ( (v43 & 1) != 0 )
    {
      v24 = &v46;
      v23 = (char *)&v44;
    }
    else
    {
      v23 = v44;
      v15 = 72LL * v45;
      if ( !v45 )
        goto LABEL_56;
      v24 = &v44[v15];
      if ( &v44[v15] == v44 )
        goto LABEL_56;
    }
    do
    {
      if ( *(_QWORD *)v23 != -4096 && *(_QWORD *)v23 != -8192 )
      {
        v25 = (char *)*((_QWORD *)v23 + 5);
        if ( v25 != v23 + 56 )
          _libc_free(v25, v15);
        v26 = (char *)*((_QWORD *)v23 + 1);
        if ( v26 != v23 + 24 )
          _libc_free(v26, v15);
      }
      v23 += 72;
    }
    while ( v24 != v23 );
    if ( (v43 & 1) != 0 )
    {
      if ( (v39 & 1) == 0 )
      {
LABEL_41:
        v27 = v40;
        v15 = 72LL * v41;
        if ( !v41 )
          goto LABEL_54;
        v28 = &v40[v15];
        if ( &v40[v15] == v40 )
          goto LABEL_54;
        goto LABEL_43;
      }
LABEL_57:
      v28 = &v42;
      v27 = (char *)&v40;
      do
      {
LABEL_43:
        if ( *(_QWORD *)v27 != -8192 && *(_QWORD *)v27 != -4096 )
        {
          v29 = (char *)*((_QWORD *)v27 + 5);
          if ( v29 != v27 + 56 )
            _libc_free(v29, v15);
          v30 = (char *)*((_QWORD *)v27 + 1);
          if ( v30 != v27 + 24 )
            _libc_free(v30, v15);
        }
        v27 += 72;
      }
      while ( v27 != v28 );
      if ( (v39 & 1) != 0 )
      {
LABEL_51:
        v16 = v35;
        result = (unsigned int)v36;
        *(_QWORD *)(a1 + 528) += (unsigned int)v36;
        if ( v16 == (unsigned __int64 *)v37 )
          goto LABEL_14;
        goto LABEL_13;
      }
      v27 = v40;
      v15 = 72LL * v41;
LABEL_54:
      sub_C7D6A0((__int64)v27, v15, 8);
      goto LABEL_51;
    }
    v23 = v44;
    v15 = 72LL * v45;
LABEL_56:
    sub_C7D6A0((__int64)v23, v15, 8);
    if ( (v39 & 1) == 0 )
      goto LABEL_41;
    goto LABEL_57;
  }
  return result;
}
