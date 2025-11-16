// Function: sub_9D45E0
// Address: 0x9d45e0
//
__int64 __fastcall sub_9D45E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const __m128i *a7,
        unsigned __int64 a8,
        __m128i a9)
{
  const __m128i *v10; // rsi
  int v11; // edx
  __int64 v12; // rcx
  char v13; // al
  unsigned __int64 v14; // rbx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rbx
  volatile signed __int32 *v22; // r13
  signed __int32 v23; // eax
  void (*v24)(); // rax
  signed __int32 v25; // eax
  __int64 (__fastcall *v26)(__int64); // rcx
  __int64 v27; // rbx
  __int64 v28; // r13
  volatile signed __int32 *v29; // r14
  signed __int32 v30; // eax
  void (*v31)(); // rax
  signed __int32 v32; // eax
  __int64 (__fastcall *v33)(__int64); // rdx
  __int8 v35; // al
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  const __m128i *v38; // rdx
  __m128i *v39; // rcx
  char v40; // al
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __int64 v43; // rax
  __int64 v44; // rcx
  char v45; // dl
  __int64 v46; // rdx
  __m128i *v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rax
  char v50; // al
  __m128i *v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // [rsp+Ch] [rbp-254h]
  __int64 v57; // [rsp+10h] [rbp-250h]
  unsigned int v58; // [rsp+18h] [rbp-248h]
  __int64 v59; // [rsp+18h] [rbp-248h]
  __int64 v60; // [rsp+20h] [rbp-240h]
  __int64 v61; // [rsp+20h] [rbp-240h]
  __int64 v62; // [rsp+20h] [rbp-240h]
  __int64 v63; // [rsp+20h] [rbp-240h]
  __int64 v64; // [rsp+28h] [rbp-238h]
  __int64 v65; // [rsp+38h] [rbp-228h] BYREF
  __int64 v66; // [rsp+40h] [rbp-220h] BYREF
  char v67; // [rsp+48h] [rbp-218h]
  const __m128i *v68; // [rsp+50h] [rbp-210h] BYREF
  __m128i *v69; // [rsp+58h] [rbp-208h]
  const __m128i *v70; // [rsp+60h] [rbp-200h]
  __m128i v71; // [rsp+68h] [rbp-1F8h] BYREF
  __m128i v72; // [rsp+78h] [rbp-1E8h] BYREF
  __m128i v73; // [rsp+90h] [rbp-1D0h] BYREF
  __m128i v74; // [rsp+A0h] [rbp-1C0h] BYREF
  __m128i v75; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i v76; // [rsp+C0h] [rbp-1A0h] BYREF
  __m128i v77; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v78; // [rsp+E0h] [rbp-180h]
  unsigned int v79; // [rsp+F0h] [rbp-170h]
  __int64 v80; // [rsp+F8h] [rbp-168h]
  __int64 v81; // [rsp+100h] [rbp-160h]
  __int64 v82; // [rsp+108h] [rbp-158h]
  __int64 v83; // [rsp+110h] [rbp-150h]
  unsigned int v84; // [rsp+118h] [rbp-148h]
  char v85; // [rsp+120h] [rbp-140h] BYREF
  char v86; // [rsp+228h] [rbp-38h]

  v10 = a7;
  sub_9D1730(&v77, (__int64)a7, a8, a4, a5);
  v11 = v86 & 1;
  v12 = (unsigned int)(2 * v11);
  v13 = (2 * v11) | v86 & 0xFD;
  v86 = v13;
  if ( (_BYTE)v11 )
  {
    *(_BYTE *)(a1 + 56) |= 3u;
    v86 = v13 & 0xFD;
    v43 = v77.m128i_i64[0];
    v77.m128i_i64[0] = 0;
    *(_QWORD *)a1 = v43 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_73;
  }
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0u;
  v72 = 0u;
  v14 = (8 * v78 - (unsigned __int64)v79) >> 3;
  if ( v77.m128i_i64[1] <= v14 + 8 )
  {
    v10 = 0;
    v39 = 0;
    v38 = 0;
LABEL_71:
    v40 = *(_BYTE *)(a1 + 56);
    v41 = _mm_loadu_si128(&v71);
    *(_QWORD *)a1 = v10;
    v42 = _mm_loadu_si128(&v72);
    *(_QWORD *)(a1 + 8) = v39;
    *(_QWORD *)(a1 + 16) = v38;
    *(__m128i *)(a1 + 24) = v41;
    *(_BYTE *)(a1 + 56) = v40 & 0xFC | 2;
    *(__m128i *)(a1 + 40) = v42;
    goto LABEL_12;
  }
  while ( 1 )
  {
    sub_9CEA50((__int64)&v66, (__int64)&v77, 0, v12);
    v15 = v67 & 1;
    v12 = (unsigned int)(2 * v15);
    v67 = (2 * v15) | v67 & 0xFD;
    if ( (_BYTE)v15 )
    {
      sub_9C9090(v73.m128i_i64, &v66);
      v49 = v73.m128i_i64[0];
      *(_BYTE *)(a1 + 56) |= 3u;
      *(_QWORD *)a1 = v49 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_8;
    }
    if ( (_DWORD)v66 == 2 )
      break;
    if ( (unsigned int)v66 <= 2 )
    {
      v73.m128i_i64[0] = (__int64)"Malformed block";
      v75.m128i_i16[0] = 259;
LABEL_7:
      sub_9C8190(&v65, (__int64)&v73);
      v16 = v65;
      *(_BYTE *)(a1 + 56) |= 3u;
      *(_QWORD *)a1 = v16 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_8;
    }
    if ( (_DWORD)v66 == 3 )
    {
      sub_A4CAE0(&v73, &v77, HIDWORD(v66));
      v35 = v73.m128i_i8[8];
      v73.m128i_i8[8] &= ~2u;
      if ( (v35 & 1) != 0 )
      {
        v36 = v73.m128i_i64[0];
        v73.m128i_i64[0] = 0;
        v65 = v36 | 1;
      }
      else
      {
        v65 = 1;
      }
      v37 = v65 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v65 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
LABEL_58:
        *(_BYTE *)(a1 + 56) |= 3u;
        *(_QWORD *)a1 = v37;
        goto LABEL_8;
      }
      goto LABEL_67;
    }
LABEL_69:
    v14 = (8 * v78 - (unsigned __int64)v79) >> 3;
    if ( v14 + 8 >= v77.m128i_i64[1] )
    {
      v38 = v70;
      v39 = v69;
      v10 = v68;
      goto LABEL_71;
    }
  }
  if ( HIDWORD(v66) == 13 )
  {
    v62 = v78;
    v58 = v79;
    sub_9CE5C0(v73.m128i_i64, (__int64)&v77, HIDWORD(v66), v12);
    v37 = v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_58;
    sub_9CEA50((__int64)&v73, (__int64)&v77, 0, v44);
    v45 = v73.m128i_i8[8] & 1;
    v73.m128i_i8[8] = (2 * (v73.m128i_i8[8] & 1)) | v73.m128i_i8[8] & 0xFD;
    if ( v45 )
    {
      sub_9C9090(&v65, v73.m128i_i64);
      v55 = v65;
      *(_BYTE *)(a1 + 56) |= 3u;
      *(_QWORD *)a1 = v55 & 0xFFFFFFFFFFFFFFFELL;
      sub_9CEF80(&v73);
      goto LABEL_8;
    }
    if ( v73.m128i_i64[0] != 0x800000002LL )
    {
      v73.m128i_i64[0] = (__int64)"Malformed block";
      v75.m128i_i16[0] = 259;
      goto LABEL_7;
    }
    v12 = v58;
    v46 = 8 * v14;
    v59 = 8 * v62 - v58 - 8 * v14;
    goto LABEL_89;
  }
  if ( HIDWORD(v66) == 8 )
  {
    v46 = 8 * v14;
    v59 = -1;
LABEL_89:
    v63 = v46;
    v57 = v78;
    v56 = v79;
    sub_9CE5C0(v73.m128i_i64, (__int64)&v77, v46, v12);
    v37 = v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_58;
    v75 = 0u;
    v47 = v69;
    v12 = v56;
    v73.m128i_i64[1] = ((8 * v78 - (unsigned __int64)v79) >> 3) - v14;
    v73.m128i_i64[0] = v77.m128i_i64[0] + v14;
    v74 = a9;
    v76.m128i_i64[0] = v59;
    v76.m128i_i64[1] = 8 * v57 - v56 - v63;
    if ( v69 == v70 )
    {
      sub_9D4420(&v68, v69, &v73);
    }
    else
    {
      if ( v69 )
      {
        *v69 = _mm_loadu_si128(&v73);
        v47[1] = _mm_loadu_si128(&v74);
        v47[2] = _mm_loadu_si128(&v75);
        v47[3] = _mm_loadu_si128(&v76);
        v47 = v69;
      }
      v69 = v47 + 4;
    }
    goto LABEL_67;
  }
  if ( HIDWORD(v66) != 23 )
  {
    if ( HIDWORD(v66) == 25 )
    {
      sub_9CF210((__int64)&v73, (__int64)&v77, 25);
      if ( (v74.m128i_i8[0] & 1) != 0 )
      {
        v48 = v73.m128i_i64[0];
        *(_BYTE *)(a1 + 56) |= 3u;
        *(_QWORD *)a1 = v48 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_8;
      }
      if ( !v71.m128i_i64[1] )
        v71 = _mm_loadu_si128(&v73);
    }
    else
    {
      sub_9CE5C0(v73.m128i_i64, (__int64)&v77, HIDWORD(v66), v12);
      v37 = v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v73.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_58;
    }
    goto LABEL_67;
  }
  sub_9CF210((__int64)&v73, (__int64)&v77, 23);
  v50 = v74.m128i_i8[0] & 1;
  v74.m128i_i8[0] = (2 * (v74.m128i_i8[0] & 1)) | v74.m128i_i8[0] & 0xFD;
  if ( !v50 )
  {
    v12 = (__int64)v68;
    v51 = v69;
    if ( v68 == v69 )
    {
      if ( !v71.m128i_i64[1] )
      {
LABEL_67:
        if ( (v67 & 2) != 0 )
          goto LABEL_84;
        if ( (v67 & 1) != 0 && v66 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v66 + 8LL))(v66);
        goto LABEL_69;
      }
      if ( v72.m128i_i64[1] )
      {
LABEL_112:
        if ( (v74.m128i_i8[0] & 1) != 0 && v73.m128i_i64[0] )
          (*(void (__fastcall **)(__int64, __m128i *, __m128i *, __int64, __m128i *))(*(_QWORD *)v73.m128i_i64[0] + 8LL))(
            v73.m128i_i64[0],
            &v77,
            v51,
            v12,
            &v73);
        goto LABEL_67;
      }
    }
    else
    {
      do
      {
        v51 -= 4;
        if ( v51[2].m128i_i64[1] )
          break;
        if ( v50 )
          goto LABEL_105;
        v51[2] = _mm_loadu_si128(&v73);
        v50 = (v74.m128i_i8[0] & 2) != 0;
      }
      while ( (__m128i *)v12 != v51 );
      if ( !v71.m128i_i64[1] || v72.m128i_i64[1] )
      {
        if ( v50 )
          goto LABEL_105;
        goto LABEL_112;
      }
      if ( v50 )
        goto LABEL_105;
    }
    v72 = _mm_loadu_si128(&v73);
    goto LABEL_112;
  }
  sub_9C98F0(&v65, v73.m128i_i64);
  v54 = v65;
  *(_BYTE *)(a1 + 56) |= 3u;
  *(_QWORD *)a1 = v54 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v74.m128i_i8[0] & 2) != 0 )
LABEL_105:
    sub_9D4340(&v73);
  if ( (v74.m128i_i8[0] & 1) != 0 && v73.m128i_i64[0] )
    (*(void (__fastcall **)(__int64, __m128i *, __int64, __int64, __m128i *))(*(_QWORD *)v73.m128i_i64[0] + 8LL))(
      v73.m128i_i64[0],
      &v73,
      v52,
      v53,
      &v73);
LABEL_8:
  if ( (v67 & 2) != 0 )
LABEL_84:
    sub_9CEF10(&v66);
  if ( (v67 & 1) != 0 && v66 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v66 + 8LL))(v66);
  v10 = (const __m128i *)((char *)v70 - (char *)v68);
  if ( v68 )
    j_j___libc_free_0(v68, v10);
LABEL_12:
  if ( (v86 & 2) != 0 )
    sub_9D03C0(&v77);
  if ( (v86 & 1) == 0 )
  {
    v17 = 32LL * v84;
    v64 = v83;
    v18 = v83 + v17;
    if ( v83 == v83 + v17 )
      goto LABEL_34;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v18 - 24);
      v20 = *(_QWORD *)(v18 - 16);
      v18 -= 32;
      v21 = v19;
      if ( v20 != v19 )
        break;
LABEL_30:
      if ( v19 )
      {
        v10 = (const __m128i *)(*(_QWORD *)(v18 + 24) - v19);
        j_j___libc_free_0(v19, v10);
      }
      if ( v64 == v18 )
      {
        v18 = v83;
LABEL_34:
        if ( (char *)v18 != &v85 )
          _libc_free(v18, v10);
        v27 = v81;
        v28 = v80;
        if ( v81 == v80 )
        {
LABEL_51:
          if ( v28 )
            j_j___libc_free_0(v28, v82 - v28);
          return a1;
        }
        while ( 1 )
        {
          v29 = *(volatile signed __int32 **)(v28 + 8);
          if ( !v29 )
            goto LABEL_38;
          if ( &_pthread_key_create )
          {
            v30 = _InterlockedExchangeAdd(v29 + 2, 0xFFFFFFFF);
          }
          else
          {
            v30 = *((_DWORD *)v29 + 2);
            *((_DWORD *)v29 + 2) = v30 - 1;
          }
          if ( v30 != 1 )
            goto LABEL_38;
          v31 = *(void (**)())(*(_QWORD *)v29 + 16LL);
          if ( v31 != nullsub_25 )
            ((void (__fastcall *)(volatile signed __int32 *, const __m128i *))v31)(v29, v10);
          if ( &_pthread_key_create )
          {
            v32 = _InterlockedExchangeAdd(v29 + 3, 0xFFFFFFFF);
          }
          else
          {
            v32 = *((_DWORD *)v29 + 3);
            *((_DWORD *)v29 + 3) = v32 - 1;
          }
          if ( v32 != 1 )
            goto LABEL_38;
          v33 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 24LL);
          if ( v33 == sub_9C26E0 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *, const __m128i *))(*(_QWORD *)v29 + 8LL))(v29, v10);
            v28 += 16;
            if ( v27 == v28 )
            {
LABEL_50:
              v28 = v80;
              goto LABEL_51;
            }
          }
          else
          {
            ((void (__fastcall *)(volatile signed __int32 *, const __m128i *))v33)(v29, v10);
LABEL_38:
            v28 += 16;
            if ( v27 == v28 )
              goto LABEL_50;
          }
        }
      }
    }
    while ( 1 )
    {
      v22 = *(volatile signed __int32 **)(v21 + 8);
      if ( !v22 )
        goto LABEL_17;
      if ( &_pthread_key_create )
      {
        v23 = _InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
      }
      else
      {
        v23 = *((_DWORD *)v22 + 2);
        *((_DWORD *)v22 + 2) = v23 - 1;
      }
      if ( v23 != 1 )
        goto LABEL_17;
      v24 = *(void (**)())(*(_QWORD *)v22 + 16LL);
      if ( v24 != nullsub_25 )
      {
        v61 = v20;
        ((void (__fastcall *)(volatile signed __int32 *, const __m128i *))v24)(v22, v10);
        v20 = v61;
      }
      if ( &_pthread_key_create )
      {
        v25 = _InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
      }
      else
      {
        v25 = *((_DWORD *)v22 + 3);
        *((_DWORD *)v22 + 3) = v25 - 1;
      }
      if ( v25 != 1 )
        goto LABEL_17;
      v60 = v20;
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 24LL);
      if ( v26 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *, const __m128i *))(*(_QWORD *)v22 + 8LL))(v22, v10);
        v20 = v60;
        v21 += 16;
        if ( v60 == v21 )
        {
LABEL_29:
          v19 = *(_QWORD *)(v18 + 8);
          goto LABEL_30;
        }
      }
      else
      {
        ((void (__fastcall *)(volatile signed __int32 *, const __m128i *))v26)(v22, v10);
        v20 = v60;
LABEL_17:
        v21 += 16;
        if ( v20 == v21 )
          goto LABEL_29;
      }
    }
  }
LABEL_73:
  if ( v77.m128i_i64[0] )
    (*(void (__fastcall **)(__int64, const __m128i *))(*(_QWORD *)v77.m128i_i64[0] + 8LL))(v77.m128i_i64[0], v10);
  return a1;
}
