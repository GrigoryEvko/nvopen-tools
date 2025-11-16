// Function: sub_248FA70
// Address: 0x248fa70
//
__int64 __fastcall sub_248FA70(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rdi
  _QWORD *v20; // r12
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  __int64 v23; // r8
  unsigned int v24; // edx
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // rdx
  int v28; // eax
  unsigned int v29; // esi
  __int64 v30; // r9
  unsigned int v31; // eax
  _QWORD *v32; // rdx
  __int64 v33; // r8
  _QWORD *v34; // r14
  unsigned __int64 v35; // rdi
  int v36; // ecx
  int v37; // r10d
  int v38; // r14d
  _QWORD *v39; // rdi
  int v40; // eax
  int v41; // eax
  __m128i *v42; // rdi
  int v43; // edx
  int v44; // r14d
  __int64 v45; // r11
  __int64 v46; // rdx
  __int64 v47; // r8
  int v48; // esi
  _QWORD *v49; // rcx
  int v50; // edx
  int v51; // r14d
  __int64 v52; // r11
  int v53; // esi
  unsigned int v54; // edx
  __int64 v55; // r8
  __int64 v56; // [rsp+10h] [rbp-100h]
  char v57; // [rsp+37h] [rbp-D9h] BYREF
  __m128i *p_s; // [rsp+38h] [rbp-D8h] BYREF
  unsigned __int64 v59[2]; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v60[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v61[8]; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD *v62; // [rsp+68h] [rbp-A8h]
  unsigned int v63; // [rsp+78h] [rbp-98h]
  char v64[8]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v65; // [rsp+88h] [rbp-88h]
  int v66; // [rsp+90h] [rbp-80h]
  unsigned int v67; // [rsp+98h] [rbp-78h]
  __m128i s; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD *v69; // [rsp+B0h] [rbp-60h]
  __int64 v70; // [rsp+B8h] [rbp-58h]
  int v71; // [rsp+C0h] [rbp-50h]
  __int64 v72; // [rsp+C8h] [rbp-48h]
  _QWORD v73[8]; // [rsp+D0h] [rbp-40h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  sub_EE18A0((__int64)v61, a3 + 160, a3, (__int64)a4, a5, a6);
  s.m128i_i64[0] = (__int64)v61;
  sub_248F0C0((__int64)v64, a2, a4, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_2485740, (__int64)&s);
  v8 = v65;
  v9 = v67;
  v10 = 3LL * v67;
  if ( v66 )
  {
    v20 = &v65[v10];
    if ( v65 != &v65[v10] )
    {
      v21 = v65;
      while ( 1 )
      {
        v22 = v21;
        if ( *v21 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v21 += 3;
        if ( v20 == v21 )
          goto LABEL_2;
      }
      if ( v20 != v21 )
      {
        while ( 1 )
        {
          if ( v63 )
          {
            v23 = v63 - 1;
            v24 = v23 & (((0xBF58476D1CE4E5B9LL * *v22) >> 31) ^ (484763065 * *(_DWORD *)v22));
            v25 = (__int64)&v62[3 * v24];
            v26 = *(_QWORD *)v25;
            if ( *v22 != *(_QWORD *)v25 )
            {
              v36 = 1;
              while ( v26 != -1 )
              {
                v37 = v36 + 1;
                v24 = v23 & (v36 + v24);
                v25 = (__int64)&v62[3 * v24];
                v26 = *(_QWORD *)v25;
                if ( *v22 == *(_QWORD *)v25 )
                  goto LABEL_24;
                v36 = v37;
              }
              goto LABEL_39;
            }
LABEL_24:
            if ( (_QWORD *)v25 != &v62[3 * v63] )
              break;
          }
LABEL_39:
          v22 += 3;
          if ( v22 != v20 )
          {
            while ( *v22 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v22 += 3;
              if ( v20 == v22 )
                goto LABEL_42;
            }
            if ( v20 != v22 )
              continue;
          }
LABEL_42:
          v8 = v65;
          v9 = v67;
          v10 = 3LL * v67;
          goto LABEL_2;
        }
        s.m128i_i64[0] = (__int64)v73;
        s.m128i_i64[1] = 1;
        v69 = 0;
        v70 = 0;
        v71 = 1065353216;
        v72 = 0;
        v73[0] = 0;
        v27 = *((unsigned int *)v22 + 4);
        p_s = &s;
        v60[1] = 0;
        v60[0] = (unsigned __int64)v61;
        if ( (_DWORD)v27 )
        {
          v56 = v25;
          sub_2484F60((__int64)v60, (__int64)(v22 + 1), v27, v25, v23, v26);
          v25 = v56;
        }
        v28 = *(_DWORD *)(v25 + 16);
        v59[0] = (unsigned __int64)v60;
        v59[1] = 0;
        if ( v28 )
          sub_2484F60((__int64)v59, v25 + 8, v27, v25, v23, v26);
        sub_248AF20(
          (__int64)v59,
          (__int64)v60,
          (unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64))sub_2484DA0,
          &v57,
          (void (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, _QWORD *))sub_248B770,
          (__int64)&p_s);
        if ( (unsigned __int64 *)v59[0] != v60 )
          _libc_free(v59[0]);
        if ( (char *)v60[0] != v61 )
          _libc_free(v60[0]);
        v29 = *(_DWORD *)(a1 + 24);
        if ( v29 )
        {
          v30 = *(_QWORD *)(a1 + 8);
          v31 = (v29 - 1) & (((0xBF58476D1CE4E5B9LL * *v22) >> 31) ^ (484763065 * *(_DWORD *)v22));
          v32 = (_QWORD *)(v30 + ((unsigned __int64)v31 << 6));
          v33 = *v32;
          if ( *v32 == *v22 )
          {
LABEL_35:
            v34 = v69;
            while ( v34 )
            {
              v35 = (unsigned __int64)v34;
              v34 = (_QWORD *)*v34;
              j_j___libc_free_0(v35);
            }
            memset((void *)s.m128i_i64[0], 0, 8 * s.m128i_i64[1]);
            v70 = 0;
            v69 = 0;
            if ( (_QWORD *)s.m128i_i64[0] != v73 )
              j_j___libc_free_0(s.m128i_u64[0]);
            goto LABEL_39;
          }
          v38 = 1;
          v39 = 0;
          while ( v33 != -1 )
          {
            if ( v39 || v33 != -2 )
              v32 = v39;
            v31 = (v29 - 1) & (v38 + v31);
            v33 = *(_QWORD *)(v30 + ((unsigned __int64)v31 << 6));
            if ( *v22 == v33 )
              goto LABEL_35;
            ++v38;
            v39 = v32;
            v32 = (_QWORD *)(v30 + ((unsigned __int64)v31 << 6));
          }
          v40 = *(_DWORD *)(a1 + 16);
          if ( !v39 )
            v39 = v32;
          ++*(_QWORD *)a1;
          v41 = v40 + 1;
          if ( 4 * v41 < 3 * v29 )
          {
            if ( v29 - *(_DWORD *)(a1 + 20) - v41 > v29 >> 3 )
              goto LABEL_55;
            sub_248BD20(a1, v29);
            v50 = *(_DWORD *)(a1 + 24);
            if ( !v50 )
            {
LABEL_84:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v51 = v50 - 1;
            v52 = *(_QWORD *)(a1 + 8);
            v53 = 1;
            v49 = 0;
            v54 = (v50 - 1) & (((0xBF58476D1CE4E5B9LL * *v22) >> 31) ^ (484763065 * *(_DWORD *)v22));
            v41 = *(_DWORD *)(a1 + 16) + 1;
            v39 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
            v55 = *v39;
            if ( *v39 == *v22 )
              goto LABEL_55;
            while ( v55 != -1 )
            {
              if ( v55 == -2 && !v49 )
                v49 = v39;
              v54 = v51 & (v53 + v54);
              v39 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
              v55 = *v39;
              if ( *v22 == *v39 )
                goto LABEL_55;
              ++v53;
            }
            goto LABEL_71;
          }
        }
        else
        {
          ++*(_QWORD *)a1;
        }
        sub_248BD20(a1, 2 * v29);
        v43 = *(_DWORD *)(a1 + 24);
        if ( !v43 )
          goto LABEL_84;
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 8);
        v46 = (v43 - 1) & ((unsigned int)((0xBF58476D1CE4E5B9LL * *v22) >> 31) ^ (484763065 * *(_DWORD *)v22));
        v41 = *(_DWORD *)(a1 + 16) + 1;
        v39 = (_QWORD *)(v45 + (v46 << 6));
        v47 = *v39;
        if ( *v22 == *v39 )
          goto LABEL_55;
        v48 = 1;
        v49 = 0;
        while ( v47 != -1 )
        {
          if ( v47 == -2 && !v49 )
            v49 = v39;
          LODWORD(v46) = v44 & (v48 + v46);
          v39 = (_QWORD *)(v45 + ((unsigned __int64)(unsigned int)v46 << 6));
          v47 = *v39;
          if ( *v22 == *v39 )
            goto LABEL_55;
          ++v48;
        }
LABEL_71:
        if ( v49 )
          v39 = v49;
LABEL_55:
        *(_DWORD *)(a1 + 16) = v41;
        if ( *v39 != -1 )
          --*(_DWORD *)(a1 + 20);
        v42 = (__m128i *)(v39 + 1);
        v42[-1].m128i_i64[1] = *v22;
        sub_248B5F0(v42, &s);
        goto LABEL_35;
      }
    }
  }
LABEL_2:
  if ( v9 )
  {
    v11 = &v8[v10];
    do
    {
      while ( 1 )
      {
        v12 = v8 + 3;
        if ( *v8 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v13 = v8[1];
          if ( (_QWORD *)v13 != v12 )
            break;
        }
        v8 = v12;
        if ( v11 == v12 )
          goto LABEL_8;
      }
      _libc_free(v13);
      v8 = v12;
    }
    while ( v11 != v12 );
LABEL_8:
    v8 = v65;
    v10 = 3LL * v67;
  }
  sub_C7D6A0((__int64)v8, v10 * 8, 8);
  v14 = v63;
  if ( v63 )
  {
    v15 = v62;
    v16 = &v62[3 * v63];
    do
    {
      while ( 1 )
      {
        v17 = v15 + 3;
        if ( *v15 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v18 = v15[1];
          if ( (_QWORD *)v18 != v17 )
            break;
        }
        v15 += 3;
        if ( v16 == v17 )
          goto LABEL_15;
      }
      _libc_free(v18);
      v15 = v17;
    }
    while ( v16 != v17 );
LABEL_15:
    v14 = v63;
  }
  sub_C7D6A0((__int64)v62, 24 * v14, 8);
  return a1;
}
