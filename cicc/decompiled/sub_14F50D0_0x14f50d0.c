// Function: sub_14F50D0
// Address: 0x14f50d0
//
__int64 __fastcall sub_14F50D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i *a7,
        unsigned __int64 a8,
        __m128i a9)
{
  __m128i *v9; // rsi
  char v10; // dl
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // r8
  unsigned __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // r12
  __int64 v21; // rbx
  volatile signed __int32 *v22; // r13
  signed __int32 v23; // eax
  signed __int32 v24; // eax
  __int64 v25; // rbx
  __int64 v26; // r13
  volatile signed __int32 *v27; // r14
  signed __int32 v28; // eax
  signed __int32 v29; // eax
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // r12
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  __m128i v36; // xmm2
  char v37; // al
  char v38; // al
  const __m128i *v39; // rcx
  unsigned __int64 v40; // rax
  unsigned int v41; // [rsp+Ch] [rbp-234h]
  unsigned __int64 v43; // [rsp+18h] [rbp-228h]
  __int64 v44; // [rsp+18h] [rbp-228h]
  __int64 v45; // [rsp+28h] [rbp-218h] BYREF
  const __m128i *v46; // [rsp+30h] [rbp-210h] BYREF
  __m128i *v47; // [rsp+38h] [rbp-208h]
  const __m128i *v48; // [rsp+40h] [rbp-200h]
  __m128i v49; // [rsp+48h] [rbp-1F8h] BYREF
  __m128i v50; // [rsp+58h] [rbp-1E8h] BYREF
  __m128i v51; // [rsp+70h] [rbp-1D0h] BYREF
  __m128i v52; // [rsp+80h] [rbp-1C0h] BYREF
  __m128i v53; // [rsp+90h] [rbp-1B0h] BYREF
  __m128i v54; // [rsp+A0h] [rbp-1A0h] BYREF
  __m128i v55; // [rsp+B0h] [rbp-190h] BYREF
  __int64 v56; // [rsp+C0h] [rbp-180h]
  unsigned int v57; // [rsp+D0h] [rbp-170h]
  __int64 v58; // [rsp+D8h] [rbp-168h]
  __int64 v59; // [rsp+E0h] [rbp-160h]
  __int64 v60; // [rsp+E8h] [rbp-158h]
  unsigned __int64 v61; // [rsp+F0h] [rbp-150h]
  unsigned int v62; // [rsp+F8h] [rbp-148h]
  char v63; // [rsp+100h] [rbp-140h] BYREF
  char v64; // [rsp+208h] [rbp-38h]

  v9 = a7;
  sub_14F3710(&v55, (__int64)a7, a8);
  v10 = v64 & 1;
  v11 = (2 * (v64 & 1)) | v64 & 0xFD;
  v64 = v11;
  if ( v10 )
  {
    v64 = v11 & 0xFD;
    v31 = v55.m128i_i64[0];
    *(_BYTE *)(a1 + 56) |= 3u;
    v55.m128i_i64[0] = 0;
    *(_QWORD *)a1 = v31 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_57;
  }
  v46 = 0;
  v12 = v56;
  v47 = 0;
  v13 = v57;
  v48 = 0;
  v49 = 0u;
  v50 = 0u;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = (unsigned __int64)(8 * v12 - v13) >> 3;
        if ( v14 + 8 >= v55.m128i_i64[1] )
        {
          v16 = (__m128i *)a1;
          v36 = _mm_loadu_si128(&v50);
          v37 = *(_BYTE *)(a1 + 56);
          *(__m128i *)(a1 + 24) = _mm_loadu_si128(&v49);
          *(__m128i *)(a1 + 40) = v36;
          *(_BYTE *)(a1 + 56) = v37 & 0xFC | 2;
          *(_QWORD *)a1 = v46;
          *(_QWORD *)(a1 + 8) = v47;
          *(_QWORD *)(a1 + 16) = v48;
          goto LABEL_9;
        }
        v15 = sub_14ECC00((__int64)&v55, 0);
        v9 = (__m128i *)HIDWORD(v15);
        if ( (_DWORD)v15 == 2 )
          break;
        if ( (unsigned int)v15 <= 2 )
          goto LABEL_6;
        if ( (_DWORD)v15 != 3 )
          goto LABEL_52;
        sub_150F8E0(&v55, v9);
        v12 = v56;
        v13 = v57;
      }
      if ( HIDWORD(v15) != 13 )
        break;
      v32 = 8 * v14;
      v33 = 8 * v56 - v57 - 8 * v14;
      if ( (unsigned __int8)sub_14ED8F0((__int64)&v55)
        || (v34 = sub_14ECC00((__int64)&v55, 0), HIDWORD(v34) != 8)
        || (_DWORD)v34 != 2 )
      {
LABEL_6:
        v9 = &v51;
        v51.m128i_i64[0] = (__int64)"Malformed block";
        v52.m128i_i16[0] = 259;
        sub_14EE0F0(&v45, (__int64)&v51);
        v16 = (__m128i *)a1;
        *(_BYTE *)(a1 + 56) |= 3u;
        *(_QWORD *)a1 = v45 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_7;
      }
LABEL_67:
      v44 = v56;
      v41 = v57;
      if ( (unsigned __int8)sub_14ED8F0((__int64)&v55) )
        goto LABEL_6;
      v12 = v56;
      v53 = 0u;
      v54.m128i_i64[0] = v33;
      v51.m128i_i64[0] = v14 + v55.m128i_i64[0];
      v13 = v57;
      v51.m128i_i64[1] = ((8 * v56 - (unsigned __int64)v57) >> 3) - v14;
      v52 = a9;
      v9 = v47;
      v54.m128i_i64[1] = 8 * v44 - v41 - v32;
      if ( v47 == v48 )
      {
        sub_14F4F10(&v46, v47, &v51);
        v12 = v56;
        v13 = v57;
      }
      else
      {
        if ( v47 )
        {
          *v47 = _mm_loadu_si128(&v51);
          v9[1] = _mm_loadu_si128(&v52);
          v12 = v56;
          v9[2] = _mm_loadu_si128(&v53);
          v13 = v57;
          v9[3] = _mm_loadu_si128(&v54);
          v9 = v47;
        }
        v9 += 4;
        v47 = v9;
      }
    }
    if ( HIDWORD(v15) == 8 )
    {
      v32 = 8 * v14;
      v33 = -1;
      goto LABEL_67;
    }
    if ( HIDWORD(v15) == 23 )
      break;
    if ( HIDWORD(v15) == 25 )
    {
      v9 = &v55;
      sub_14EEBA0((__int64)&v51, (__int64)&v55, 0x19u);
      if ( (v52.m128i_i8[0] & 1) != 0 )
      {
        v16 = (__m128i *)a1;
        v35 = v51.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
        *(_BYTE *)(a1 + 56) |= 3u;
        *(_QWORD *)a1 = v35;
        goto LABEL_7;
      }
      if ( !v49.m128i_i64[1] )
        v49 = _mm_loadu_si128(&v51);
    }
    else if ( (unsigned __int8)sub_14ED8F0((__int64)&v55) )
    {
      goto LABEL_6;
    }
LABEL_52:
    v12 = v56;
    v13 = v57;
  }
  v9 = &v55;
  sub_14EEBA0((__int64)&v51, (__int64)&v55, 0x17u);
  v38 = v52.m128i_i8[0] & 1;
  v52.m128i_i8[0] = (2 * (v52.m128i_i8[0] & 1)) | v52.m128i_i8[0] & 0xFD;
  if ( !v38 )
  {
    v16 = v47;
    v39 = v46;
    if ( v46 == v47 )
    {
      if ( !v49.m128i_i64[1] )
        goto LABEL_52;
      if ( v50.m128i_i64[1] )
        goto LABEL_85;
LABEL_84:
      v50 = _mm_loadu_si128(&v51);
    }
    else
    {
      while ( !v16[-2].m128i_i64[1] )
      {
        if ( v38 )
          goto LABEL_80;
        v16 -= 4;
        v16[2] = _mm_loadu_si128(&v51);
        v38 = (v52.m128i_i8[0] & 2) != 0;
        if ( v39 == v16 )
          break;
      }
      if ( v49.m128i_i64[1] && !v50.m128i_i64[1] )
      {
        if ( v38 )
          goto LABEL_80;
        goto LABEL_84;
      }
      if ( v38 )
        goto LABEL_80;
    }
LABEL_85:
    if ( (v52.m128i_i8[0] & 1) != 0 && v51.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v51.m128i_i64[0] + 8LL))(v51.m128i_i64[0]);
    goto LABEL_52;
  }
  v9 = &v51;
  sub_14EF680(&v45, v51.m128i_i64);
  v16 = (__m128i *)a1;
  v40 = v45 & 0xFFFFFFFFFFFFFFFELL;
  *(_BYTE *)(a1 + 56) |= 3u;
  *(_QWORD *)a1 = v40;
  if ( (v52.m128i_i8[0] & 2) != 0 )
LABEL_80:
    sub_14F4E30(&v51, (__int64)v9, (__int64)v16);
  if ( (v52.m128i_i8[0] & 1) != 0 && v51.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v51.m128i_i64[0] + 8LL))(v51.m128i_i64[0]);
LABEL_7:
  if ( v46 )
  {
    v9 = (__m128i *)((char *)v48 - (char *)v46);
    j_j___libc_free_0(v46, (char *)v48 - (char *)v46);
  }
LABEL_9:
  if ( (v64 & 2) != 0 )
    sub_14F2A80(&v55, (__int64)v9, (__int64)v16);
  if ( (v64 & 1) != 0 )
  {
LABEL_57:
    if ( v55.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v55.m128i_i64[0] + 8LL))(v55.m128i_i64[0]);
    return a1;
  }
  v17 = 32LL * v62;
  v43 = v61;
  v18 = v61 + v17;
  if ( v61 != v61 + v17 )
  {
    do
    {
      v19 = *(_QWORD *)(v18 - 24);
      v20 = *(_QWORD *)(v18 - 16);
      v18 -= 32LL;
      v21 = v19;
      if ( v20 != v19 )
      {
        do
        {
          while ( 1 )
          {
            v22 = *(volatile signed __int32 **)(v21 + 8);
            if ( v22 )
            {
              if ( &_pthread_key_create )
              {
                v23 = _InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
              }
              else
              {
                v23 = *((_DWORD *)v22 + 2);
                *((_DWORD *)v22 + 2) = v23 - 1;
              }
              if ( v23 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
                if ( &_pthread_key_create )
                {
                  v24 = _InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v24 = *((_DWORD *)v22 + 3);
                  *((_DWORD *)v22 + 3) = v24 - 1;
                }
                if ( v24 == 1 )
                  break;
              }
            }
            v21 += 16;
            if ( v20 == v21 )
              goto LABEL_23;
          }
          v21 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
        }
        while ( v20 != v21 );
LABEL_23:
        v19 = *(_QWORD *)(v18 + 8);
      }
      if ( v19 )
        j_j___libc_free_0(v19, *(_QWORD *)(v18 + 24) - v19);
    }
    while ( v43 != v18 );
    v18 = v61;
  }
  if ( (char *)v18 != &v63 )
    _libc_free(v18);
  v25 = v59;
  v26 = v58;
  if ( v59 != v58 )
  {
    do
    {
      while ( 1 )
      {
        v27 = *(volatile signed __int32 **)(v26 + 8);
        if ( v27 )
        {
          if ( &_pthread_key_create )
          {
            v28 = _InterlockedExchangeAdd(v27 + 2, 0xFFFFFFFF);
          }
          else
          {
            v28 = *((_DWORD *)v27 + 2);
            *((_DWORD *)v27 + 2) = v28 - 1;
          }
          if ( v28 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 16LL))(v27);
            if ( &_pthread_key_create )
            {
              v29 = _InterlockedExchangeAdd(v27 + 3, 0xFFFFFFFF);
            }
            else
            {
              v29 = *((_DWORD *)v27 + 3);
              *((_DWORD *)v27 + 3) = v29 - 1;
            }
            if ( v29 == 1 )
              break;
          }
        }
        v26 += 16;
        if ( v25 == v26 )
          goto LABEL_41;
      }
      v26 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 24LL))(v27);
    }
    while ( v25 != v26 );
LABEL_41:
    v26 = v58;
  }
  if ( v26 )
    j_j___libc_free_0(v26, v60 - v26);
  return a1;
}
