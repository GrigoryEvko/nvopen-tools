// Function: sub_9D1140
// Address: 0x9d1140
//
void (__fastcall *__fastcall sub_9D1140(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7))(__int64, __int64, __int64)
{
  __int32 v11; // r8d
  __int64 v12; // rdi
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  __int32 v15; // edx
  __int64 v16; // rax
  __m128i v17; // xmm1
  __m128i v18; // xmm0
  _BYTE *v19; // r14
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  volatile signed __int32 *v22; // r15
  signed __int32 v23; // eax
  void (*v24)(); // rax
  signed __int32 v25; // eax
  __int64 (__fastcall *v26)(__int64); // rdx
  __m128i v27; // xmm2
  void (__fastcall *result)(__int64, __int64, __int64); // rax
  __int64 v29; // r8
  _BYTE *v30; // r12
  __int64 v31; // rdi
  __int64 v32; // r15
  __int64 v33; // rbx
  volatile signed __int32 *v34; // rdi
  signed __int32 v35; // eax
  void (*v36)(void); // rax
  signed __int32 v37; // eax
  void (*v38)(void); // rdx
  __m128i v41; // [rsp+30h] [rbp-1C0h]
  __m128i v42; // [rsp+40h] [rbp-1B0h]
  __m128i v43; // [rsp+60h] [rbp-190h] BYREF
  __m128i v44; // [rsp+70h] [rbp-180h] BYREF
  __int32 v45; // [rsp+80h] [rbp-170h]
  __int32 v46; // [rsp+84h] [rbp-16Ch]
  _QWORD *v47; // [rsp+88h] [rbp-168h]
  _QWORD *v48; // [rsp+90h] [rbp-160h]
  __int64 v49; // [rsp+98h] [rbp-158h]
  _BYTE *v50; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v51; // [rsp+A8h] [rbp-148h]
  _BYTE v52[256]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v53; // [rsp+1B0h] [rbp-40h]

  v41 = _mm_loadu_si128(a2);
  v50 = v52;
  v42 = _mm_loadu_si128(a2 + 1);
  v11 = a2[2].m128i_i32[1];
  v12 = a2[2].m128i_i64[1];
  v13 = a2[3].m128i_i64[1];
  v45 = a2[2].m128i_i32[0];
  v14 = a2[3].m128i_u64[0];
  v46 = v11;
  v47 = (_QWORD *)v12;
  v48 = (_QWORD *)v14;
  v49 = v13;
  a2[3].m128i_i64[1] = 0;
  a2[3].m128i_i64[0] = 0;
  a2[2].m128i_i64[1] = 0;
  v51 = 0x800000000LL;
  v43 = v41;
  v44 = v42;
  v15 = a2[4].m128i_i32[2];
  if ( v15 )
  {
    sub_9D06B0((__int64)&v50, (__int64)a2[4].m128i_i64);
    v11 = v46;
    v12 = (__int64)v47;
    v14 = (unsigned __int64)v48;
    v13 = v49;
    v15 = v51;
  }
  v16 = a2[21].m128i_i64[0];
  *(_DWORD *)(a1 + 60) = v11;
  v17 = _mm_loadu_si128(&v43);
  v18 = _mm_loadu_si128(&v44);
  *(_QWORD *)a1 = 0;
  v53 = v16;
  LODWORD(v16) = v45;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 56) = v16;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = v12;
  *(_QWORD *)(a1 + 72) = v14;
  *(_QWORD *)(a1 + 80) = v13;
  v49 = 0;
  v48 = 0;
  v47 = 0;
  *(_QWORD *)(a1 + 96) = 0x800000000LL;
  *(__m128i *)(a1 + 24) = v17;
  *(__m128i *)(a1 + 40) = v18;
  if ( !v15 )
  {
    *(_QWORD *)(a1 + 368) = a3;
    v19 = v50;
    *(_QWORD *)(a1 + 376) = a4;
    *(_BYTE *)(a1 + 384) = 0;
    *(_QWORD *)(a1 + 392) = a1 + 408;
    *(_QWORD *)(a1 + 400) = 0;
    *(_BYTE *)(a1 + 408) = 0;
    *(_QWORD *)(a1 + 360) = a1;
    goto LABEL_5;
  }
  v14 = (unsigned __int64)&v50;
  sub_9D06B0(a1 + 88, (__int64)&v50);
  v29 = (unsigned int)v51;
  *(_QWORD *)(a1 + 368) = a3;
  v30 = v50;
  *(_QWORD *)(a1 + 376) = a4;
  v29 *= 32;
  *(_BYTE *)(a1 + 384) = 0;
  v19 = &v30[v29];
  *(_QWORD *)(a1 + 392) = a1 + 408;
  *(_QWORD *)(a1 + 400) = 0;
  *(_BYTE *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 360) = a1;
  if ( v30 != &v30[v29] )
  {
    while ( 1 )
    {
      v31 = *((_QWORD *)v19 - 3);
      v32 = *((_QWORD *)v19 - 2);
      v19 -= 32;
      v33 = v31;
      if ( v32 != v31 )
        break;
LABEL_44:
      if ( v31 )
      {
        v14 = *((_QWORD *)v19 + 3) - v31;
        j_j___libc_free_0(v31, v14);
      }
      if ( v30 == v19 )
      {
        v19 = v50;
        goto LABEL_5;
      }
    }
    while ( 1 )
    {
      v34 = *(volatile signed __int32 **)(v33 + 8);
      if ( !v34 )
        goto LABEL_31;
      if ( &_pthread_key_create )
      {
        v35 = _InterlockedExchangeAdd(v34 + 2, 0xFFFFFFFF);
      }
      else
      {
        v35 = *((_DWORD *)v34 + 2);
        v14 = (unsigned int)(v35 - 1);
        *((_DWORD *)v34 + 2) = v14;
      }
      if ( v35 != 1 )
        goto LABEL_31;
      v36 = *(void (**)(void))(*(_QWORD *)v34 + 16LL);
      if ( v36 != nullsub_25 )
        v36();
      if ( &_pthread_key_create )
      {
        v37 = _InterlockedExchangeAdd(v34 + 3, 0xFFFFFFFF);
      }
      else
      {
        v37 = *((_DWORD *)v34 + 3);
        *((_DWORD *)v34 + 3) = v37 - 1;
      }
      if ( v37 != 1 )
        goto LABEL_31;
      v38 = *(void (**)(void))(*(_QWORD *)v34 + 24LL);
      if ( (char *)v38 == (char *)sub_9C26E0 )
      {
        (*(void (**)(void))(*(_QWORD *)v34 + 8LL))();
        v33 += 16;
        if ( v32 == v33 )
        {
LABEL_43:
          v31 = *((_QWORD *)v19 + 1);
          goto LABEL_44;
        }
      }
      else
      {
        v38();
LABEL_31:
        v33 += 16;
        if ( v32 == v33 )
          goto LABEL_43;
      }
    }
  }
LABEL_5:
  if ( v19 != v52 )
    _libc_free(v19, v14);
  v20 = v48;
  v21 = v47;
  if ( v48 != v47 )
  {
    while ( 1 )
    {
      v22 = (volatile signed __int32 *)v21[1];
      if ( !v22 )
        goto LABEL_9;
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
        goto LABEL_9;
      v24 = *(void (**)())(*(_QWORD *)v22 + 16LL);
      if ( v24 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v24)(v22);
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
        goto LABEL_9;
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 24LL);
      if ( v26 == sub_9C26E0 )
      {
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 8LL))(v22);
        v21 += 2;
        if ( v20 == v21 )
        {
LABEL_21:
          v21 = v47;
          break;
        }
      }
      else
      {
        v26((__int64)v22);
LABEL_9:
        v21 += 2;
        if ( v20 == v21 )
          goto LABEL_21;
      }
    }
  }
  if ( v21 )
    j_j___libc_free_0(v21, v49 - (_QWORD)v21);
  v27 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 424) = a5;
  *(_WORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_DWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = a1 + 528;
  *(_QWORD *)(a1 + 520) = 0;
  *(_BYTE *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(__m128i *)(a1 + 544) = v27;
  result = *(void (__fastcall **)(__int64, __int64, __int64))(a6 + 16);
  if ( result )
  {
    result(a1 + 560, a6, 2);
    *(_QWORD *)(a1 + 584) = *(_QWORD *)(a6 + 24);
    result = *(void (__fastcall **)(__int64, __int64, __int64))(a6 + 16);
    *(_QWORD *)(a1 + 576) = result;
  }
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  return result;
}
