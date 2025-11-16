// Function: sub_3909790
// Address: 0x3909790
//
__int64 __fastcall sub_3909790(unsigned int *a1, __int64 a2, __int64 a3, unsigned __int64 a4, unsigned __int64 a5)
{
  int v8; // r9d
  unsigned int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // rax
  void *v12; // rdi
  int v13; // r14d
  __int64 v15; // rax
  _DWORD *v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // r14
  _DWORD *v19; // rbx
  int v20; // eax
  unsigned __int64 v21; // r12
  __m128i v22; // xmm0
  bool v23; // cc
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  _DWORD *v27; // rax
  unsigned __int64 v28; // rdi
  int *v29; // r15
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 v35; // r12
  __m128i v36; // xmm4
  int v37; // eax
  unsigned __int64 v38; // rax
  int v39; // edx
  unsigned __int64 v40; // rdi
  __int64 v41; // rdx
  size_t v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rdi
  unsigned int v46; // eax
  __int64 v47; // rax
  unsigned __int64 v48; // [rsp+0h] [rbp-E0h]
  int v49; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v50; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int64 v51; // [rsp+28h] [rbp-B8h] BYREF
  unsigned int v52; // [rsp+30h] [rbp-B0h]
  char v53; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v54; // [rsp+40h] [rbp-A0h]
  void *src; // [rsp+48h] [rbp-98h] BYREF
  size_t n; // [rsp+50h] [rbp-90h]
  _BYTE v57[64]; // [rsp+58h] [rbp-88h] BYREF
  __m128i v58[4]; // [rsp+98h] [rbp-48h] BYREF

  v54 = a2;
  n = 0x4000000000LL;
  src = v57;
  v58[0] = 0u;
  sub_16E2F40(a3, (__int64)&src);
  v58[0] = (__m128i)__PAIR128__(a5, a4);
  v9 = a1[8];
  if ( v9 >= a1[9] )
  {
    sub_3909530(a1 + 6, 0);
    v9 = a1[8];
  }
  v10 = *((_QWORD *)a1 + 3) + 104LL * v9;
  if ( v10 )
  {
    v11 = v54;
    v12 = (void *)(v10 + 24);
    *(_QWORD *)(v10 + 8) = v10 + 24;
    *(_QWORD *)v10 = v11;
    *(_QWORD *)(v10 + 16) = 0x4000000000LL;
    v13 = n;
    if ( (void **)(v10 + 8) != &src && (_DWORD)n )
    {
      v42 = (unsigned int)n;
      if ( (unsigned int)n <= 0x40
        || (sub_16CD150(v10 + 8, (const void *)(v10 + 24), (unsigned int)n, 1, v10 + 8, v8),
            v42 = (unsigned int)n,
            v12 = *(void **)(v10 + 8),
            (_DWORD)n) )
      {
        memcpy(v12, src, v42);
      }
      *(_DWORD *)(v10 + 16) = v13;
    }
    *(__m128i *)(v10 + 88) = _mm_loadu_si128(v58);
    v9 = a1[8];
  }
  a1[8] = v9 + 1;
  if ( *(_DWORD *)sub_3909460((__int64)a1) == 1 )
  {
    v15 = (*(__int64 (__fastcall **)(unsigned int *))(*(_QWORD *)a1 + 40LL))(a1);
    v16 = *(_DWORD **)(v15 + 8);
    v17 = *(unsigned int *)(v15 + 16);
    v18 = v15;
    v19 = v16 + 10;
    *(_BYTE *)(v15 + 114) = *v16 == 9;
    v20 = v17;
    v17 *= 40LL;
    v21 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v17 - 40) >> 3);
    if ( v17 > 0x28 )
    {
      do
      {
        v22 = _mm_loadu_si128((const __m128i *)(v19 + 2));
        v23 = *(v19 - 2) <= 0x40u;
        *(v19 - 10) = *v19;
        *((__m128i *)v19 - 2) = v22;
        if ( !v23 )
        {
          v24 = *((_QWORD *)v19 - 2);
          if ( v24 )
            j_j___libc_free_0_0(v24);
        }
        v25 = *((_QWORD *)v19 + 3);
        v19 += 10;
        *((_QWORD *)v19 - 7) = v25;
        LODWORD(v25) = *(v19 - 2);
        *(v19 - 2) = 0;
        *(v19 - 12) = v25;
        --v21;
      }
      while ( v21 );
      v20 = *(_DWORD *)(v18 + 16);
      v16 = *(_DWORD **)(v18 + 8);
    }
    v26 = (unsigned int)(v20 - 1);
    *(_DWORD *)(v18 + 16) = v26;
    v27 = &v16[10 * v26];
    if ( v27[8] > 0x40u )
    {
      v28 = *((_QWORD *)v27 + 3);
      if ( v28 )
        j_j___libc_free_0_0(v28);
    }
    if ( !*(_DWORD *)(v18 + 16) )
    {
      v29 = &v49;
      (**(void (__fastcall ***)(int *, __int64))v18)(&v49, v18);
      v30 = *(unsigned int *)(v18 + 16);
      v31 = *(_QWORD *)(v18 + 8);
      LODWORD(v32) = *(_DWORD *)(v18 + 16);
      v33 = 40 * v30;
      v34 = v31 + 40 * v30;
      if ( 40 * v30 )
      {
        if ( v30 >= *(unsigned int *)(v18 + 20) )
        {
          sub_38E8F60(v18 + 8, 0);
          v31 = *(_QWORD *)(v18 + 8);
          v32 = *(unsigned int *)(v18 + 16);
          v33 = 40 * v32;
          v34 = v31 + 40 * v32;
        }
        v35 = v31 + v33 - 40;
        if ( v34 )
        {
          v36 = _mm_loadu_si128((const __m128i *)(v35 + 8));
          *(_DWORD *)v34 = *(_DWORD *)v35;
          *(__m128i *)(v34 + 8) = v36;
          v37 = *(_DWORD *)(v35 + 32);
          *(_DWORD *)(v35 + 32) = 0;
          *(_DWORD *)(v34 + 32) = v37;
          *(_QWORD *)(v34 + 24) = *(_QWORD *)(v35 + 24);
          v32 = *(unsigned int *)(v18 + 16);
          v34 = *(_QWORD *)(v18 + 8) + 40 * v32;
          v35 = v34 - 40;
        }
        v38 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v35 - v31) >> 3);
        if ( (__int64)(v35 - v31) > 0 )
        {
          do
          {
            v39 = *(_DWORD *)(v35 - 40);
            v34 -= 40;
            v35 -= 40;
            v23 = *(_DWORD *)(v34 + 32) <= 0x40u;
            *(_DWORD *)v34 = v39;
            *(__m128i *)(v34 + 8) = _mm_loadu_si128((const __m128i *)(v35 + 8));
            if ( !v23 )
            {
              v40 = *(_QWORD *)(v34 + 24);
              if ( v40 )
              {
                v48 = v38;
                j_j___libc_free_0_0(v40);
                v38 = v48;
              }
            }
            *(_QWORD *)(v34 + 24) = *(_QWORD *)(v35 + 24);
            *(_DWORD *)(v34 + 32) = *(_DWORD *)(v35 + 32);
            *(_DWORD *)(v35 + 32) = 0;
            --v38;
          }
          while ( v38 );
          LODWORD(v32) = *(_DWORD *)(v18 + 16);
        }
        v41 = (unsigned int)(v32 + 1);
        *(_DWORD *)(v18 + 16) = v41;
        if ( v31 <= (unsigned __int64)&v49 && (unsigned __int64)&v49 < *(_QWORD *)(v18 + 8) + 40 * v41 )
          v29 = (int *)&v53;
        v23 = *(_DWORD *)(v31 + 32) <= 0x40u;
        *(_DWORD *)v31 = *v29;
        *(__m128i *)(v31 + 8) = _mm_loadu_si128((const __m128i *)(v29 + 2));
        if ( v23 && (unsigned int)v29[8] <= 0x40 )
        {
          v43 = *((_QWORD *)v29 + 3);
          *(_QWORD *)(v31 + 24) = v43;
          v44 = (unsigned int)v29[8];
          *(_DWORD *)(v31 + 32) = v44;
          v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
          if ( (unsigned int)v44 > 0x40 )
          {
            v47 = (unsigned int)((unsigned __int64)(v44 + 63) >> 6) - 1;
            *(_QWORD *)(v43 + 8 * v47) &= v45;
          }
          else
          {
            *(_QWORD *)(v31 + 24) = v45 & v43;
          }
        }
        else
        {
          sub_16A51C0(v31 + 24, (__int64)(v29 + 6));
        }
      }
      else
      {
        if ( (unsigned int)v30 >= *(_DWORD *)(v18 + 20) )
        {
          sub_38E8F60(v18 + 8, 0);
          LODWORD(v32) = *(_DWORD *)(v18 + 16);
          v34 = *(_QWORD *)(v18 + 8) + 40LL * (unsigned int)v32;
        }
        if ( v34 )
        {
          *(_DWORD *)v34 = v49;
          *(__m128i *)(v34 + 8) = _mm_loadu_si128(&v50);
          v46 = v52;
          *(_DWORD *)(v34 + 32) = v52;
          if ( v46 > 0x40 )
            sub_16A4FD0(v34 + 24, (const void **)&v51);
          else
            *(_QWORD *)(v34 + 24) = v51;
          LODWORD(v32) = *(_DWORD *)(v18 + 16);
        }
        *(_DWORD *)(v18 + 16) = v32 + 1;
      }
      if ( v52 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
    }
  }
  if ( src != v57 )
    _libc_free((unsigned __int64)src);
  return 1;
}
