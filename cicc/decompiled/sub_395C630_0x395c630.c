// Function: sub_395C630
// Address: 0x395c630
//
__int64 __fastcall sub_395C630(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        unsigned int *a7)
{
  unsigned int v7; // r14d
  unsigned __int8 v8; // al
  __int64 *v9; // rbx
  __int64 v11; // r15
  unsigned int v13; // eax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rsi
  __int64 *v17; // rsi
  int v18; // r8d
  unsigned __int8 v19; // r10
  unsigned int *v20; // r9
  __int64 v21; // r11
  __int64 v22; // rax
  const __m128i *v23; // r9
  __m128i *v24; // r8
  __int64 v25; // rdx
  const __m128i *v26; // r14
  unsigned __int8 v27; // r15
  const __m128i *v28; // rbx
  __int64 v29; // rcx
  __m128i *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r15
  __int64 v33; // rbx
  __int32 v34; // eax
  int v35; // r8d
  int v36; // r9d
  __int32 v37; // ecx
  __int64 v38; // rsi
  __int64 v39; // rax
  __m128i *v40; // rax
  __int64 v41; // rdx
  int v42; // r15d
  __int64 v43; // r12
  __int64 v44; // r14
  unsigned int v46; // eax
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rax
  __m128i *v50; // rax
  __int64 v51; // rdx
  unsigned int v52; // eax
  int v53; // r8d
  int v54; // r9d
  __int64 v55; // rax
  __m128i *v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // eax
  int v59; // r8d
  int v60; // r9d
  unsigned int v61; // eax
  __m128i *v62; // rax
  __int64 *v63; // [rsp+18h] [rbp-238h]
  __int64 v64; // [rsp+18h] [rbp-238h]
  __int64 v65; // [rsp+28h] [rbp-228h]
  unsigned __int8 v66; // [rsp+28h] [rbp-228h]
  unsigned int v67; // [rsp+30h] [rbp-220h]
  unsigned int v68; // [rsp+34h] [rbp-21Ch]
  unsigned __int8 v70; // [rsp+38h] [rbp-218h]
  unsigned __int8 v71; // [rsp+50h] [rbp-200h]
  __int64 v72; // [rsp+50h] [rbp-200h]
  __int64 *v73; // [rsp+50h] [rbp-200h]
  int v74; // [rsp+58h] [rbp-1F8h]
  unsigned __int8 v75; // [rsp+58h] [rbp-1F8h]
  __m128i v77; // [rsp+60h] [rbp-1F0h] BYREF
  __int64 v78; // [rsp+70h] [rbp-1E0h]
  const __m128i *v79; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v80; // [rsp+88h] [rbp-1C8h]
  _BYTE v81[192]; // [rsp+90h] [rbp-1C0h] BYREF
  __m128i v82; // [rsp+150h] [rbp-100h] BYREF
  _QWORD v83[30]; // [rsp+160h] [rbp-F0h] BYREF

  v7 = 0;
  if ( *(_BYTE *)(*a2 + 8) != 11 )
    return v7;
  v8 = *((_BYTE *)a2 + 16);
  v9 = a2;
  if ( v8 <= 0x17u || (unsigned int)v8 - 35 > 0x11 )
    return v7;
  v11 = a1;
  v74 = v8 - 24;
  if ( v8 != 39 )
  {
    if ( ((v8 - 35) & 0xFD) != 0 )
      return v7;
    v13 = *a7;
    if ( *a7 )
    {
      v14 = a2[1];
      if ( !v14 || *(_QWORD *)(v14 + 8) )
        return v7;
    }
    v7 = 0;
    if ( v13 >= dword_5054AE0 )
      return v7;
    v15 = (int)a2;
    *a7 = v13 + 1;
    v16 = *(a2 - 6);
    v80 = 0x800000000LL;
    v65 = v16;
    v79 = (const __m128i *)v81;
    v71 = sub_395C630(a1, v16, v15, (unsigned int)&v79, a5, (_DWORD)a6, (__int64)a7);
    v68 = *(_DWORD *)(a5 + 8);
    v17 = (__int64 *)*(v9 - 3);
    v82.m128i_i64[0] = (__int64)v83;
    v82.m128i_i64[1] = 0x800000000LL;
    v7 = v71;
    v19 = sub_395C630(a1, (_DWORD)v17, (_DWORD)v9, (unsigned int)&v82, a5, (_DWORD)a6, (__int64)a7);
    LOBYTE(v7) = v19 | v71;
    if ( !(v19 | v71) )
      goto LABEL_30;
    v20 = a6;
    v21 = v65;
    v67 = *(_DWORD *)(a5 + 8);
    v22 = a6[2];
    if ( (unsigned int)v22 >= a6[3] )
    {
      v64 = v65;
      v66 = v19;
      sub_16CD150((__int64)a6, a6 + 4, 0, 8, v18, (int)a6);
      v20 = a6;
      v21 = v64;
      v19 = v66;
      v22 = a6[2];
    }
    *(_QWORD *)(*(_QWORD *)v20 + 8 * v22) = v9;
    ++v20[2];
    if ( v71 )
    {
      v23 = (const __m128i *)((char *)v79 + 24 * (unsigned int)v80);
      if ( v79 != v23 )
      {
        v24 = &v77;
        v25 = *(unsigned int *)(a4 + 8);
        v70 = v7;
        v26 = (const __m128i *)((char *)v79 + 24 * (unsigned int)v80);
        v27 = v19;
        v63 = v9;
        v28 = v79;
        do
        {
          v29 = v28[1].m128i_i64[0];
          v77 = _mm_loadu_si128(v28);
          v78 = v29;
          if ( *(_DWORD *)(a4 + 12) <= (unsigned int)v25 )
          {
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 24, (int)v24, (int)v23);
            v25 = *(unsigned int *)(a4 + 8);
          }
          v28 = (const __m128i *)((char *)v28 + 24);
          v30 = (__m128i *)(*(_QWORD *)a4 + 24 * v25);
          v31 = v78;
          *v30 = _mm_loadu_si128(&v77);
          v30[1].m128i_i64[0] = v31;
          v25 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
          *(_DWORD *)(a4 + 8) = v25;
        }
        while ( v26 != v28 );
        v19 = v27;
        v7 = v70;
        v11 = a1;
        v9 = v63;
      }
      if ( v19 )
      {
LABEL_21:
        v32 = v82.m128i_i64[0];
        v72 = v82.m128i_i64[0] + 24LL * v82.m128i_u32[2];
        if ( v82.m128i_i64[0] != v72 )
        {
          do
          {
            v33 = *(_QWORD *)(v32 + 16);
            v34 = sub_395C610(v74, *(_DWORD *)(v32 + 12));
            v37 = *(_DWORD *)(v32 + 8);
            v38 = *(_QWORD *)v32;
            v78 = v33;
            v77.m128i_i32[3] = v34;
            v39 = *(unsigned int *)(a4 + 8);
            v77.m128i_i64[0] = v38;
            v77.m128i_i32[2] = v37;
            if ( (unsigned int)v39 >= *(_DWORD *)(a4 + 12) )
            {
              sub_16CD150(a4, (const void *)(a4 + 16), 0, 24, v35, v36);
              v39 = *(unsigned int *)(a4 + 8);
            }
            v32 += 24;
            v40 = (__m128i *)(*(_QWORD *)a4 + 24 * v39);
            v41 = v78;
            *v40 = _mm_loadu_si128(&v77);
            v40[1].m128i_i64[0] = v41;
            ++*(_DWORD *)(a4 + 8);
          }
          while ( v72 != v32 );
          v7 = (unsigned __int8)v7;
        }
        if ( v67 > v68 )
        {
          v42 = v74;
          v75 = v7;
          v43 = 24LL * v68;
          do
          {
            v44 = v43 + *(_QWORD *)a5;
            v43 += 24;
            *(_DWORD *)(v44 + 12) = sub_395C610(v42, *(_DWORD *)(v44 + 12));
          }
          while ( v43 != 24 * (v68 + (unsigned __int64)(v67 - 1 - v68) + 1) );
          v7 = v75;
        }
        goto LABEL_30;
      }
      if ( (unsigned int)sub_14C23D0((__int64)v17, v11, 0, 0, 0, 0) )
      {
        v46 = sub_3959780(v11, v17);
        if ( v46 )
        {
          v77.m128i_i64[1] = __PAIR64__(v74, v46);
          v77.m128i_i64[0] = (__int64)v17;
          v49 = *(unsigned int *)(a5 + 8);
          v78 = (__int64)v9;
          if ( (unsigned int)v49 >= *(_DWORD *)(a5 + 12) )
          {
            sub_16CD150(a5, (const void *)(a5 + 16), 0, 24, v47, v48);
            v49 = *(unsigned int *)(a5 + 8);
          }
          v7 = v71;
          v50 = (__m128i *)(*(_QWORD *)a5 + 24 * v49);
          v51 = v78;
          *v50 = _mm_loadu_si128(&v77);
          v50[1].m128i_i64[0] = v51;
          ++*(_DWORD *)(a5 + 8);
          goto LABEL_30;
        }
      }
    }
    else
    {
      v73 = (__int64 *)v21;
      if ( (unsigned int)sub_14C23D0(v21, a1, 0, 0, 0, 0) )
      {
        v52 = sub_3959780(a1, v73);
        if ( v52 )
        {
          v77.m128i_i64[1] = v52 | 0xB00000000LL;
          v55 = *(unsigned int *)(a5 + 8);
          v77.m128i_i64[0] = (__int64)v73;
          v78 = (__int64)v9;
          if ( (unsigned int)v55 >= *(_DWORD *)(a5 + 12) )
          {
            sub_16CD150(a5, (const void *)(a5 + 16), 0, 24, v53, v54);
            v55 = *(unsigned int *)(a5 + 8);
          }
          v56 = (__m128i *)(*(_QWORD *)a5 + 24 * v55);
          v57 = v78;
          *v56 = _mm_loadu_si128(&v77);
          v56[1].m128i_i64[0] = v57;
          ++*(_DWORD *)(a5 + 8);
          goto LABEL_21;
        }
      }
    }
    v7 = 0;
LABEL_30:
    if ( (_QWORD *)v82.m128i_i64[0] != v83 )
      _libc_free(v82.m128i_u64[0]);
    if ( v79 != (const __m128i *)v81 )
      _libc_free((unsigned __int64)v79);
    return v7;
  }
  if ( *(_BYTE *)(*(_QWORD *)*(a2 - 6) + 8LL) == 11 && *(_BYTE *)(*(_QWORD *)*(a2 - 3) + 8LL) == 11 )
  {
    if ( (unsigned int)sub_14C23D0((__int64)a2, a1, 0, 0, 0, 0) && (v58 = sub_3959780(a1, a2)) != 0 )
    {
      v82.m128i_i64[1] = v58 | 0xB00000000LL;
      v82.m128i_i64[0] = (__int64)a2;
      v61 = *(_DWORD *)(a4 + 12);
      v83[0] = a3;
      if ( *(_DWORD *)(a4 + 8) >= v61 )
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 24, v59, v60);
      v7 = 1;
      v62 = (__m128i *)(*(_QWORD *)a4 + 24LL * *(unsigned int *)(a4 + 8));
      *v62 = _mm_loadu_si128(&v82);
      v62[1].m128i_i64[0] = v83[0];
      ++*(_DWORD *)(a4 + 8);
    }
    else
    {
      return 0;
    }
  }
  return v7;
}
