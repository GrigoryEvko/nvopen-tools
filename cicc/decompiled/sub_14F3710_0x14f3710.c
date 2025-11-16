// Function: sub_14F3710
// Address: 0x14f3710
//
__m128i *__fastcall sub_14F3710(__m128i *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // r8
  unsigned int v5; // eax
  __int64 v6; // rdx
  char v7; // cl
  int v8; // edx
  int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rbx
  volatile signed __int32 *v17; // r12
  signed __int32 v18; // eax
  signed __int32 v19; // eax
  __int64 v20; // rbx
  __int64 v21; // r12
  volatile signed __int32 *v22; // r14
  signed __int32 v23; // eax
  signed __int32 v24; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __m128i v28; // xmm1
  __m128i v29; // xmm0
  __int32 v30; // eax
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v34; // [rsp+18h] [rbp-1F8h]
  __int64 v35; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 v36[2]; // [rsp+60h] [rbp-1B0h] BYREF
  char v37; // [rsp+70h] [rbp-1A0h]
  char v38; // [rsp+71h] [rbp-19Fh]
  __m128i v39; // [rsp+80h] [rbp-190h] BYREF
  __m128i v40; // [rsp+90h] [rbp-180h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-170h]
  __int64 v42; // [rsp+A8h] [rbp-168h]
  __int64 v43; // [rsp+B0h] [rbp-160h]
  __int64 v44; // [rsp+B8h] [rbp-158h]
  _BYTE *v45; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v46; // [rsp+C8h] [rbp-148h]
  _BYTE v47[256]; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v48; // [rsp+1D0h] [rbp-40h]

  v3 = a3 & 3;
  if ( (a3 & 3) != 0 )
  {
    v39.m128i_i64[0] = (__int64)"Invalid bitcode signature";
    v40.m128i_i16[0] = 259;
    sub_14EE0F0(v36, (__int64)&v39);
    v26 = v36[0] & 0xFFFFFFFFFFFFFFFELL;
    a1[21].m128i_i8[8] |= 3u;
    a1->m128i_i64[0] = v26;
  }
  else
  {
    if ( a2 == a2 + a3 )
    {
      a3 = 0;
      LODWORD(v4) = 0;
    }
    else if ( *(_BYTE *)a2 == 0xDE && *(_BYTE *)(a2 + 1) == 0xC0 && *(_BYTE *)(a2 + 2) == 23 && *(_BYTE *)(a2 + 3) == 11 )
    {
      if ( (unsigned int)a3 <= 0xF || (v27 = *(unsigned int *)(a2 + 8), v4 = *(unsigned int *)(a2 + 12), a3 < v4 + v27) )
      {
        v39.m128i_i64[0] = (__int64)"Invalid bitcode wrapper header";
        v40.m128i_i16[0] = 259;
        sub_14EE0F0(v36, (__int64)&v39);
        v32 = v36[0] & 0xFFFFFFFFFFFFFFFELL;
        a1[21].m128i_i8[8] |= 3u;
        a1->m128i_i64[0] = v32;
        return a1;
      }
      a2 += v27;
      a3 = v4;
    }
    else
    {
      LODWORD(v4) = a3;
    }
    v39.m128i_i64[0] = a2;
    v41 = 0x200000000LL;
    v45 = v47;
    v39.m128i_i64[1] = a3;
    v40 = 0u;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v46 = 0x800000000LL;
    v48 = 0;
    if ( a3 <= 3 )
      goto LABEL_12;
    if ( a3 > 7 )
    {
      v3 = *(_QWORD *)a2;
      v9 = 56;
      v8 = 64;
      v40.m128i_i64[0] = 8;
    }
    else
    {
      v5 = 0;
      do
      {
        v6 = v5;
        v7 = 8 * v5++;
        v3 |= (unsigned __int64)*(unsigned __int8 *)(a2 + v6) << v7;
      }
      while ( (_DWORD)v4 != v5 );
      v8 = 8 * v4;
      v40.m128i_i64[0] = (unsigned int)v4;
      v9 = 8 * v4 - 8;
    }
    LODWORD(v41) = v9;
    v40.m128i_i64[1] = v3 >> 8;
    if ( (_BYTE)v3 == 66
      && (LODWORD(v41) = v8 - 16, v40.m128i_i64[1] = v3 >> 16, BYTE1(v3) == 67)
      && (v40.m128i_i64[1] = v3 >> 20, LODWORD(v41) = v8 - 20, ((v3 >> 16) & 0xF) == 0)
      && sub_14ECAB0((__int64)&v39, 4u) == 12
      && sub_14ECAB0((__int64)&v39, 4u) == 14
      && sub_14ECAB0((__int64)&v39, 4u) == 13 )
    {
      v28 = _mm_loadu_si128(&v39);
      v29 = _mm_loadu_si128(&v40);
      a1[21].m128i_i8[8] = a1[21].m128i_i8[8] & 0xFC | 2;
      a1[2].m128i_i32[0] = v41;
      v30 = HIDWORD(v41);
      *a1 = v28;
      a1[2].m128i_i32[1] = v30;
      v31 = v42;
      a1[1] = v29;
      a1[2].m128i_i64[1] = v31;
      v42 = 0;
      a1[3].m128i_i64[0] = v43;
      v43 = 0;
      a1[3].m128i_i64[1] = v44;
      a1[4].m128i_i64[0] = (__int64)a1[5].m128i_i64;
      a1[4].m128i_i64[1] = 0x800000000LL;
      v11 = (unsigned int)v46;
      v44 = 0;
      if ( (_DWORD)v46 )
      {
        sub_14F2DD0((__int64)a1[4].m128i_i64, (__int64 *)&v45);
        v11 = (unsigned int)v46;
      }
      a1[21].m128i_i64[0] = v48;
    }
    else
    {
LABEL_12:
      v38 = 1;
      v36[0] = (__int64)"Invalid bitcode signature";
      v37 = 3;
      sub_14EE0F0(&v35, (__int64)v36);
      v10 = v35;
      a1[21].m128i_i8[8] |= 3u;
      a1->m128i_i64[0] = v10 & 0xFFFFFFFFFFFFFFFELL;
      v11 = (unsigned int)v46;
    }
    v12 = 32 * v11;
    v13 = (unsigned __int64)&v45[v12];
    v34 = (__int64)v45;
    if ( v45 != &v45[v12] )
    {
      do
      {
        v14 = *(_QWORD *)(v13 - 24);
        v15 = *(_QWORD *)(v13 - 16);
        v13 -= 32LL;
        v16 = v14;
        if ( v15 != v14 )
        {
          do
          {
            while ( 1 )
            {
              v17 = *(volatile signed __int32 **)(v16 + 8);
              if ( v17 )
              {
                if ( &_pthread_key_create )
                {
                  v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v18 = *((_DWORD *)v17 + 2);
                  *((_DWORD *)v17 + 2) = v18 - 1;
                }
                if ( v18 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 16LL))(v17);
                  if ( &_pthread_key_create )
                  {
                    v19 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v19 = *((_DWORD *)v17 + 3);
                    *((_DWORD *)v17 + 3) = v19 - 1;
                  }
                  if ( v19 == 1 )
                    break;
                }
              }
              v16 += 16;
              if ( v15 == v16 )
                goto LABEL_25;
            }
            v16 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
          }
          while ( v15 != v16 );
LABEL_25:
          v14 = *(_QWORD *)(v13 + 8);
        }
        if ( v14 )
          j_j___libc_free_0(v14, *(_QWORD *)(v13 + 24) - v14);
      }
      while ( v34 != v13 );
      v13 = (unsigned __int64)v45;
    }
    if ( (_BYTE *)v13 != v47 )
      _libc_free(v13);
    v20 = v43;
    v21 = v42;
    if ( v43 != v42 )
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
            goto LABEL_43;
        }
        v21 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
      }
      while ( v20 != v21 );
LABEL_43:
      v21 = v42;
    }
    if ( v21 )
      j_j___libc_free_0(v21, v44 - v21);
  }
  return a1;
}
