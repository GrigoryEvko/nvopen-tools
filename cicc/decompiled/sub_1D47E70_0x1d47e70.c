// Function: sub_1D47E70
// Address: 0x1d47e70
//
__int64 *__fastcall sub_1D47E70(_DWORD *a1, __m128 *a2, __m128i a3, double a4, __m128i a5)
{
  bool v6; // zf
  __int64 *v7; // r15
  __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 *v12; // r8
  __int64 *v13; // r15
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // rcx
  int i; // eax
  __int64 v19; // rdx
  unsigned int v20; // eax
  unsigned int v21; // eax
  __int64 v22; // r15
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 *v25; // r15
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // r15
  unsigned __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rsi
  __int128 v33; // [rsp-10h] [rbp-1D0h]
  __int64 v34; // [rsp+0h] [rbp-1C0h]
  __int64 v35; // [rsp+0h] [rbp-1C0h]
  __int64 v36; // [rsp+8h] [rbp-1B8h]
  __int64 *v37; // [rsp+18h] [rbp-1A8h]
  __m128 v38; // [rsp+20h] [rbp-1A0h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-190h] BYREF
  __int64 (__fastcall *v40)(__m128i **, const __m128i **, int); // [rsp+40h] [rbp-180h]
  __int64 *(__fastcall *v41)(__int64 **, __int64 *, __int64, __int64, __int64, int); // [rsp+48h] [rbp-178h]
  __int64 *v42; // [rsp+50h] [rbp-170h] BYREF
  __int64 v43; // [rsp+58h] [rbp-168h]
  _BYTE v44[48]; // [rsp+60h] [rbp-160h] BYREF
  _BYTE *v45; // [rsp+90h] [rbp-130h] BYREF
  __int64 v46; // [rsp+98h] [rbp-128h]
  _BYTE v47[64]; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v48; // [rsp+E0h] [rbp-E0h] BYREF
  __m128 *v49; // [rsp+E8h] [rbp-D8h]
  void *s; // [rsp+F0h] [rbp-D0h]
  _BYTE v51[12]; // [rsp+F8h] [rbp-C8h]
  _BYTE v52[184]; // [rsp+108h] [rbp-B8h] BYREF

  v49 = (__m128 *)v52;
  v6 = a1[2] == 1;
  s = v52;
  v46 = 0x800000000LL;
  v42 = (__int64 *)v44;
  v37 = (__int64 *)a2;
  v48 = 0;
  *(_QWORD *)v51 = 16;
  *(_DWORD *)&v51[8] = 0;
  v45 = v47;
  v43 = 0x300000000LL;
  if ( v6 )
  {
    v7 = **(__int64 ***)(**(_QWORD **)a1 + 32LL);
    goto LABEL_3;
  }
  v9 = 24;
  v40 = 0;
  v10 = (_QWORD *)sub_22077B0(24);
  if ( v10 )
  {
    *v10 = &v48;
    v10[1] = v39;
    v10[2] = &v42;
  }
  v39[0] = v10;
  v12 = *(__int64 **)a1;
  v41 = sub_1D477A0;
  v40 = sub_1D46700;
  v13 = v12;
  v14 = &v12[a1[2]];
  if ( v14 != v12 )
  {
    do
    {
LABEL_12:
      v11 = *v13;
      v15 = (unsigned int)v46;
      if ( (unsigned int)v46 >= HIDWORD(v46) )
      {
        v34 = *v13;
        sub_16CD150((__int64)&v45, v47, 0, 8, (int)v12, v11);
        v15 = (unsigned int)v46;
        v11 = v34;
      }
      *(_QWORD *)&v45[8 * v15] = v11;
      v16 = (__int64 *)v49;
      LODWORD(v46) = v46 + 1;
      if ( s != v49 )
        goto LABEL_10;
      a2 = (__m128 *)((char *)v49 + 8 * *(unsigned int *)&v51[4]);
      v9 = *(unsigned int *)&v51[4];
      if ( v49 != a2 )
      {
        v17 = 0;
        while ( v11 != *v16 )
        {
          if ( *v16 == -2 )
            v17 = v16;
          if ( a2 == (__m128 *)++v16 )
          {
            if ( !v17 )
              goto LABEL_33;
            ++v13;
            *v17 = v11;
            --*(_DWORD *)&v51[8];
            ++v48;
            if ( v14 != v13 )
              goto LABEL_12;
            goto LABEL_23;
          }
        }
        goto LABEL_11;
      }
LABEL_33:
      if ( *(_DWORD *)&v51[4] < *(_DWORD *)v51 )
      {
        v9 = (unsigned int)++*(_DWORD *)&v51[4];
        a2->m128_u64[0] = v11;
        ++v48;
      }
      else
      {
LABEL_10:
        a2 = (__m128 *)v11;
        v9 = (__int64)&v48;
        sub_16CCBA0((__int64)&v48, v11);
      }
LABEL_11:
      ++v13;
    }
    while ( v14 != v13 );
  }
LABEL_23:
  for ( i = v46; (_DWORD)v46; i = v46 )
  {
    v19 = *(_QWORD *)&v45[8 * i - 8];
    LODWORD(v46) = i - 1;
    a3 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v19 + 32));
    v38 = (__m128)a3;
    if ( !v40 )
      sub_4263D6(v9, a2, v19);
    v9 = (__int64)v39;
    a2 = &v38;
    ((void (__fastcall *)(_QWORD *, __m128 *))v41)(v39, &v38);
  }
  v20 = v43;
  v7 = v37 + 11;
  if ( (_DWORD)v43 )
  {
    ++v48;
    if ( s != v49 )
    {
      v21 = 4 * (*(_DWORD *)&v51[4] - *(_DWORD *)&v51[8]);
      if ( v21 < 0x20 )
        v21 = 32;
      if ( *(_DWORD *)v51 > v21 )
      {
        sub_16CC920((__int64)&v48);
        v20 = v43;
LABEL_41:
        v22 = v20;
        v23 = v42;
        v24 = (unsigned int)v46;
        v25 = &v42[2 * v22];
        if ( v25 != v42 )
        {
          do
          {
            v26 = *v23;
            if ( HIDWORD(v46) <= (unsigned int)v24 )
            {
              v35 = *v23;
              sub_16CD150((__int64)&v45, v47, 0, 8, v26, v11);
              v24 = (unsigned int)v46;
              v26 = v35;
            }
            v23 += 2;
            *(_QWORD *)&v45[8 * v24] = v26;
            v24 = (unsigned int)(v46 + 1);
            LODWORD(v46) = v46 + 1;
          }
          while ( v25 != v23 );
        }
        v27 = *(_QWORD *)a1;
        v28 = *(__int64 **)a1;
        v36 = *(_QWORD *)a1 + 8LL * (unsigned int)a1[2];
        if ( v36 == *(_QWORD *)a1 )
        {
LABEL_50:
          if ( (unsigned int)v43 == 1 )
          {
            v7 = (__int64 *)*v42;
          }
          else
          {
            v29 = (unsigned __int64)v42;
            v30 = (unsigned int)v43;
            v31 = **(_QWORD **)a1;
            v32 = *(_QWORD *)(v31 + 72);
            v38.m128_u64[0] = v32;
            if ( v32 )
              sub_1623A60((__int64)&v38, v32, 2);
            *((_QWORD *)&v33 + 1) = v30;
            *(_QWORD *)&v33 = v29;
            v38.m128_i32[2] = *(_DWORD *)(v31 + 64);
            v7 = sub_1D359D0(v37, 2, (__int64)&v38, 1, 0, 0, *(double *)a3.m128i_i64, a4, a5, v33);
            if ( v38.m128_u64[0] )
              sub_161E7C0((__int64)&v38, v38.m128_i64[0]);
          }
        }
        else
        {
          while ( !(unsigned __int8)sub_1D15B50(*v28, (__int64)&v48, (__int64)&v45, 0x2000u, 1, v27) )
          {
            if ( (__int64 *)v36 == ++v28 )
              goto LABEL_50;
          }
          v7 = 0;
        }
        goto LABEL_27;
      }
      memset(s, -1, 8LL * *(unsigned int *)v51);
      v20 = v43;
    }
    *(_QWORD *)&v51[4] = 0;
    goto LABEL_41;
  }
LABEL_27:
  if ( v40 )
    v40((__m128i **)v39, (const __m128i **)v39, 3);
  if ( v42 != (__int64 *)v44 )
    _libc_free((unsigned __int64)v42);
  if ( v45 != v47 )
    _libc_free((unsigned __int64)v45);
LABEL_3:
  if ( s != v49 )
    _libc_free((unsigned __int64)s);
  return v7;
}
