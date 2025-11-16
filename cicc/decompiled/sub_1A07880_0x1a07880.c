// Function: sub_1A07880
// Address: 0x1a07880
//
__int64 __fastcall sub_1A07880(__int64 a1, __int64 a2, __int64 a3, __m128 a4, __m128 a5, double a6, __int64 a7, int a8)
{
  __int64 v8; // r15
  unsigned int v9; // ecx
  char *v11; // rsi
  unsigned int v14; // r11d
  unsigned int v15; // eax
  __int64 v16; // r10
  unsigned int v17; // edi
  char *v18; // rdx
  unsigned int v19; // r9d
  __int64 v20; // rdi
  unsigned int i; // r14d
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // r10
  int v25; // edx
  unsigned int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // r15
  __int64 v29; // r10
  _QWORD *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  char *v33; // rcx
  size_t v34; // r15
  char *v35; // rax
  __int64 v36; // rdi
  __m128i *v37; // r14
  const __m128i *v38; // r9
  __int64 v39; // rcx
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rcx
  char *v43; // r11
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  char *v47; // r11
  __int64 v48; // rax
  __int64 v49; // rcx
  __m128i *v50; // r8
  int v51; // r9d
  __int64 v52; // rsi
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  char v55; // al
  int v56; // eax
  __int32 v57; // eax
  __int64 v58; // r9
  char *v59; // r11
  __int64 v60; // r12
  __m128i *v61; // r15
  __m128i *v62; // rax
  unsigned __int64 v63; // r9
  __int64 v64; // r10
  char *v65; // r11
  __m128i *v66; // rbx
  const __m128i *v67; // rax
  __int64 v68; // rax
  __int8 *v70; // rbx
  __m128i *v71; // [rsp+0h] [rbp-100h]
  __int64 v72; // [rsp+0h] [rbp-100h]
  __int64 v73; // [rsp+8h] [rbp-F8h]
  char *v74; // [rsp+8h] [rbp-F8h]
  __int64 v75; // [rsp+8h] [rbp-F8h]
  __m128i v76; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v77; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+40h] [rbp-C0h]
  __int64 v79; // [rsp+48h] [rbp-B8h]
  __int64 v80; // [rsp+50h] [rbp-B0h]
  int v81; // [rsp+58h] [rbp-A8h]
  __int64 v82; // [rsp+60h] [rbp-A0h]
  __int64 v83; // [rsp+68h] [rbp-98h]
  void *src; // [rsp+80h] [rbp-80h] BYREF
  __int64 v85; // [rsp+88h] [rbp-78h]
  _BYTE v86[112]; // [rsp+90h] [rbp-70h] BYREF

  v8 = 0;
  v9 = *(_DWORD *)(a3 + 8);
  if ( v9 > 3 )
  {
    v11 = *(char **)a3;
    v14 = 0;
    src = v86;
    v85 = 0x400000000LL;
    v15 = 1;
    do
    {
      v16 = *(_QWORD *)&v11[16 * v15 - 8];
      if ( v9 <= v15 )
      {
        v19 = v15;
      }
      else
      {
        v17 = 1;
        v18 = &v11[16 * v15];
        while ( 1 )
        {
          v19 = v15++;
          if ( v16 != *((_QWORD *)v18 + 1) )
            break;
          ++v17;
          v18 += 16;
          if ( v9 == v15 )
          {
            v19 = v9;
            break;
          }
        }
        if ( v17 > 1 )
          v14 += v17;
      }
      v15 = v19 + 1;
    }
    while ( v9 > v19 + 1 );
    v8 = 0;
    if ( v14 > 3 )
    {
      v20 = 0;
      for ( i = 1; i < v9; ++i )
      {
        v22 = *(_QWORD *)&v11[16 * i - 8];
        v23 = i;
        if ( i >= v9 )
        {
LABEL_58:
          v9 = *(_DWORD *)(a3 + 8);
        }
        else
        {
          v24 = i + 1;
          v25 = 1;
          v26 = v9 + 1 - i;
          while ( 1 )
          {
            i = v24 - 1;
            if ( v22 != *(_QWORD *)&v11[16 * v23 + 8] )
            {
              if ( v25 == 1 )
                goto LABEL_58;
              goto LABEL_18;
            }
            ++v25;
            v23 = v24;
            if ( v25 == v26 )
              break;
            ++v24;
          }
          i = v24;
          if ( v25 == 1 )
            goto LABEL_58;
LABEL_18:
          v27 = v25 & 0xFFFFFFFE;
          i -= v27;
          v28 = v27;
          v29 = v27;
          if ( (unsigned int)v20 >= HIDWORD(v85) )
          {
            v72 = v27;
            v75 = v22;
            sub_16CD150((__int64)&src, v86, 0, 16, a8, v22);
            v20 = (unsigned int)v85;
            v29 = v72;
            v22 = v75;
          }
          v30 = (char *)src + 16 * v20;
          v30[1] = v29;
          *v30 = v22;
          v11 = *(char **)a3;
          v20 = (unsigned int)(v85 + 1);
          v31 = 16 * (v28 + i);
          v32 = 16LL * *(unsigned int *)(a3 + 8);
          LODWORD(v85) = v85 + 1;
          v33 = &v11[16 * i];
          v34 = v32 - v31;
          if ( &v11[v31] != &v11[v32] )
          {
            v35 = (char *)memmove(&v11[16 * i], &v11[v31], v34);
            v11 = *(char **)a3;
            v20 = (unsigned int)v85;
            v33 = v35;
          }
          *(_DWORD *)(a3 + 8) = (&v33[v34] - v11) >> 4;
          v9 = (&v33[v34] - v11) >> 4;
        }
      }
      v36 = 16 * v20;
      v37 = (__m128i *)src;
      v38 = (const __m128i *)((char *)src + v36);
      v39 = v36 >> 4;
      if ( v36 )
      {
        while ( 1 )
        {
          v71 = (__m128i *)v38;
          v40 = 16 * v39;
          v73 = v39;
          v41 = sub_2207800(16 * v39, &unk_435FF63);
          v42 = v73;
          v38 = v71;
          v43 = (char *)v41;
          if ( v41 )
            break;
          v39 = v73 >> 1;
          if ( !(v73 >> 1) )
            goto LABEL_63;
        }
        a5 = (__m128)_mm_loadu_si128(v37);
        v44 = v41 + v40;
        v45 = v41 + 16;
        *(__m128 *)(v45 - 16) = a5;
        if ( v44 == v45 )
        {
          v46 = (__int64)v43;
        }
        else
        {
          do
          {
            a4 = (__m128)_mm_loadu_si128((const __m128i *)(v45 - 16));
            v45 += 16;
            *(__m128 *)(v45 - 16) = a4;
          }
          while ( v44 != v45 );
          v46 = (__int64)&v43[v40 - 16];
        }
        v74 = v43;
        v37->m128i_i64[0] = *(_QWORD *)v46;
        v37->m128i_i32[2] = *(_DWORD *)(v46 + 8);
        sub_1A01380(v37, v71, v43, v42);
        v47 = v74;
      }
      else
      {
LABEL_63:
        v40 = 0;
        sub_1A003F0(v37, v38);
        v47 = 0;
      }
      j_j___libc_free_0(v47, v40);
      v48 = sub_16498A0(a2);
      v52 = *(_QWORD *)(a2 + 48);
      v77.m128i_i64[0] = 0;
      v79 = v48;
      v53 = *(_QWORD *)(a2 + 40);
      v80 = 0;
      v77.m128i_i64[1] = v53;
      v81 = 0;
      v82 = 0;
      v83 = 0;
      v78 = a2 + 24;
      v76.m128i_i64[0] = v52;
      if ( v52 )
      {
        sub_1623A60((__int64)&v76, v52, 2);
        v50 = &v76;
        if ( v77.m128i_i64[0] )
        {
          sub_161E7C0((__int64)&v77, v77.m128i_i64[0]);
          v54 = (unsigned __int8 *)v76.m128i_i64[0];
          v50 = &v76;
        }
        else
        {
          v54 = (unsigned __int8 *)v76.m128i_i64[0];
        }
        v77.m128i_i64[0] = (__int64)v54;
        if ( v54 )
          sub_1623210((__int64)&v76, v54, (__int64)&v77);
      }
      v55 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      if ( v55 == 16 )
        v55 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
      if ( (unsigned __int8)(v55 - 1) <= 5u || *(_BYTE *)(a2 + 16) == 76 )
      {
        v56 = *(_BYTE *)(a2 + 17) >> 1;
        if ( v56 == 127 )
          v56 = -1;
        v81 = v56;
      }
      v8 = sub_1A073B0(
             a1,
             (__int64)&v77,
             (__int64)&src,
             *(double *)a4.m128_u64,
             *(double *)a5.m128_u64,
             a6,
             v49,
             (__int64)v50,
             v51);
      if ( *(_DWORD *)(a3 + 8) )
      {
        v57 = sub_1A03A70(a1, v8);
        v58 = *(unsigned int *)(a3 + 8);
        v59 = *(char **)a3;
        v76.m128i_i64[1] = v8;
        v60 = 16 * v58;
        v76.m128i_i32[0] = v57;
        v61 = (__m128i *)&v59[16 * v58];
        v62 = (__m128i *)sub_19FECB0(v59, (__int64)v61, &v76);
        v66 = v62;
        if ( v61 == v62 )
        {
          if ( (unsigned int)v63 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)&v76, v63);
            v66 = (__m128i *)(*(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8));
          }
          v8 = 0;
          *v66 = _mm_load_si128(&v76);
          ++*(_DWORD *)(a3 + 8);
        }
        else
        {
          if ( v63 >= *(unsigned int *)(a3 + 12) )
          {
            v70 = (__int8 *)((char *)v62 - v65);
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)&v76, v63);
            v65 = *(char **)a3;
            v64 = *(unsigned int *)(a3 + 8);
            v66 = (__m128i *)&v70[*(_QWORD *)a3];
            v60 = 16 * v64;
            v61 = (__m128i *)(*(_QWORD *)a3 + 16 * v64);
          }
          v67 = (const __m128i *)&v65[v60 - 16];
          if ( v61 )
          {
            *v61 = _mm_loadu_si128(v67);
            v65 = *(char **)a3;
            v64 = *(unsigned int *)(a3 + 8);
            v60 = 16 * v64;
            v67 = (const __m128i *)(*(_QWORD *)a3 + 16 * v64 - 16);
          }
          if ( v66 != v67 )
          {
            memmove(&v65[v60 - ((char *)v67 - (char *)v66)], v66, (char *)v67 - (char *)v66);
            LODWORD(v64) = *(_DWORD *)(a3 + 8);
          }
          v68 = (unsigned int)(v64 + 1);
          *(_DWORD *)(a3 + 8) = v68;
          if ( v66 > &v76 || (unsigned __int64)&v76 >= *(_QWORD *)a3 + 16 * v68 )
          {
            v8 = 0;
            *v66 = _mm_load_si128(&v76);
          }
          else
          {
            v8 = 0;
            *v66 = _mm_load_si128(&v77);
          }
        }
      }
      if ( v77.m128i_i64[0] )
        sub_161E7C0((__int64)&v77, v77.m128i_i64[0]);
      if ( src != v86 )
        _libc_free((unsigned __int64)src);
    }
  }
  return v8;
}
