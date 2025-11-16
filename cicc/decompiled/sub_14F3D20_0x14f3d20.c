// Function: sub_14F3D20
// Address: 0x14f3d20
//
__int64 __fastcall sub_14F3D20(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int128 a7)
{
  const __m128i *v7; // rax
  __int32 v11; // edx
  __int32 v12; // r8d
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int32 v16; // edx
  __int64 v17; // rax
  __m128i v18; // xmm1
  __m128i v19; // xmm0
  unsigned __int64 v20; // r13
  __int64 v21; // r13
  __int64 v22; // r12
  volatile signed __int32 *v23; // r15
  signed __int32 v24; // eax
  signed __int32 v25; // eax
  __m128i v26; // xmm2
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // r15
  volatile signed __int32 *v33; // r12
  signed __int32 v34; // edx
  signed __int32 v35; // edx
  __int64 v38; // [rsp+20h] [rbp-1D0h]
  const __m128i *v39; // [rsp+28h] [rbp-1C8h]
  __m128i v40; // [rsp+60h] [rbp-190h] BYREF
  __m128i v41; // [rsp+70h] [rbp-180h] BYREF
  __int32 v42; // [rsp+80h] [rbp-170h]
  __int32 v43; // [rsp+84h] [rbp-16Ch]
  __int64 v44; // [rsp+88h] [rbp-168h]
  __int64 v45; // [rsp+90h] [rbp-160h]
  __int64 v46; // [rsp+98h] [rbp-158h]
  _BYTE *v47; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-148h]
  _BYTE v49[256]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v50; // [rsp+1B0h] [rbp-40h]

  v7 = a2;
  v40 = _mm_loadu_si128(a2);
  v11 = a2[2].m128i_i32[0];
  v41 = _mm_loadu_si128(a2 + 1);
  v12 = a2[2].m128i_i32[1];
  v13 = a2[2].m128i_i64[1];
  v14 = a2[3].m128i_i64[1];
  v42 = v11;
  v15 = a2[3].m128i_i64[0];
  v47 = v49;
  v43 = v12;
  v44 = v13;
  v45 = v15;
  v46 = v14;
  v7[3].m128i_i64[1] = 0;
  v7[3].m128i_i64[0] = 0;
  v7[2].m128i_i64[1] = 0;
  v48 = 0x800000000LL;
  v16 = v7[4].m128i_i32[2];
  if ( v16 )
  {
    v39 = v7;
    sub_14F2DD0((__int64)&v47, v7[4].m128i_i64);
    v12 = v43;
    v13 = v44;
    v15 = v45;
    v14 = v46;
    v16 = v48;
    v7 = v39;
  }
  v17 = v7[21].m128i_i64[0];
  *(_DWORD *)(a1 + 60) = v12;
  v18 = _mm_loadu_si128(&v40);
  v19 = _mm_loadu_si128(&v41);
  *(_QWORD *)a1 = 0;
  v50 = v17;
  LODWORD(v17) = v42;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 56) = v17;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 64) = v13;
  *(_QWORD *)(a1 + 72) = v15;
  *(_QWORD *)(a1 + 80) = v14;
  v46 = 0;
  v45 = 0;
  v44 = 0;
  *(_QWORD *)(a1 + 96) = 0x800000000LL;
  *(__m128i *)(a1 + 24) = v18;
  *(__m128i *)(a1 + 40) = v19;
  if ( v16 )
  {
    sub_14F2DD0(a1 + 88, (__int64 *)&v47);
    v28 = (unsigned int)v48;
    *(_QWORD *)(a1 + 376) = a4;
    *(_QWORD *)(a1 + 392) = a1 + 408;
    v29 = (__int64)v47;
    v28 *= 32;
    *(_QWORD *)(a1 + 368) = a3;
    v20 = v29 + v28;
    *(_BYTE *)(a1 + 384) = 0;
    *(_QWORD *)(a1 + 400) = 0;
    *(_BYTE *)(a1 + 408) = 0;
    *(_QWORD *)(a1 + 360) = a1;
    v38 = v29;
    if ( v29 != v29 + v28 )
    {
      do
      {
        v30 = *(_QWORD *)(v20 - 24);
        v31 = *(_QWORD *)(v20 - 16);
        v20 -= 32LL;
        v32 = v30;
        if ( v31 != v30 )
        {
          do
          {
            while ( 1 )
            {
              v33 = *(volatile signed __int32 **)(v32 + 8);
              if ( v33 )
              {
                if ( &_pthread_key_create )
                {
                  v34 = _InterlockedExchangeAdd(v33 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v34 = *((_DWORD *)v33 + 2);
                  *((_DWORD *)v33 + 2) = v34 - 1;
                }
                if ( v34 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 16LL))(v33);
                  if ( &_pthread_key_create )
                  {
                    v35 = _InterlockedExchangeAdd(v33 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v35 = *((_DWORD *)v33 + 3);
                    *((_DWORD *)v33 + 3) = v35 - 1;
                  }
                  if ( v35 == 1 )
                    break;
                }
              }
              v32 += 16;
              if ( v31 == v32 )
                goto LABEL_35;
            }
            v32 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 24LL))(v33);
          }
          while ( v31 != v32 );
LABEL_35:
          v30 = *(_QWORD *)(v20 + 8);
        }
        if ( v30 )
          j_j___libc_free_0(v30, *(_QWORD *)(v20 + 24) - v30);
      }
      while ( v38 != v20 );
      v20 = (unsigned __int64)v47;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 368) = a3;
    *(_BYTE *)(a1 + 384) = 0;
    *(_QWORD *)(a1 + 392) = a1 + 408;
    *(_QWORD *)(a1 + 400) = 0;
    *(_BYTE *)(a1 + 408) = 0;
    *(_QWORD *)(a1 + 360) = a1;
    *(_QWORD *)(a1 + 376) = a4;
    v20 = (unsigned __int64)v47;
  }
  if ( (_BYTE *)v20 != v49 )
    _libc_free(v20);
  v21 = v45;
  v22 = v44;
  if ( v45 != v44 )
  {
    do
    {
      while ( 1 )
      {
        v23 = *(volatile signed __int32 **)(v22 + 8);
        if ( v23 )
        {
          if ( &_pthread_key_create )
          {
            v24 = _InterlockedExchangeAdd(v23 + 2, 0xFFFFFFFF);
          }
          else
          {
            v24 = *((_DWORD *)v23 + 2);
            *((_DWORD *)v23 + 2) = v24 - 1;
          }
          if ( v24 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v23 + 16LL))(v23);
            if ( &_pthread_key_create )
            {
              v25 = _InterlockedExchangeAdd(v23 + 3, 0xFFFFFFFF);
            }
            else
            {
              v25 = *((_DWORD *)v23 + 3);
              *((_DWORD *)v23 + 3) = v25 - 1;
            }
            if ( v25 == 1 )
              break;
          }
        }
        v22 += 16;
        if ( v21 == v22 )
          goto LABEL_18;
      }
      v22 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v23 + 24LL))(v23);
    }
    while ( v21 != v22 );
LABEL_18:
    v22 = v44;
  }
  if ( v22 )
    j_j___libc_free_0(v22, v46 - v22);
  v26 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 424) = a5;
  *(_WORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 512) = a1 + 528;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_DWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_BYTE *)(a1 + 528) = 0;
  *(_DWORD *)(a1 + 560) = a6;
  *(__m128i *)(a1 + 544) = v26;
  return a6;
}
