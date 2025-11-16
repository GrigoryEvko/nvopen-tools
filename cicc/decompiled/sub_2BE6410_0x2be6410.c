// Function: sub_2BE6410
// Address: 0x2be6410
//
unsigned __int64 **__fastcall sub_2BE6410(unsigned __int64 **a1, _QWORD *a2)
{
  _QWORD *v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rdx
  unsigned __int64 *v7; // r14
  const __m128i *v8; // r8
  __m128i v9; // xmm6
  __m128i v10; // xmm7
  __int64 v11; // rdx
  __int64 v12; // rdx
  __m128i *v13; // rsi
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int64 v21; // r14
  int *v22; // rax
  int *v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int32 v26; // eax
  int *v27; // rax
  int *v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __int64 i; // r9
  __int64 v33; // rdi
  __int64 v34; // r10
  int *v35; // rax
  int *v36; // r8
  __int64 v37; // rsi
  __int64 v38; // rdx
  int *v39; // rsi
  __int64 v40; // r8
  int *v41; // rax
  int *v42; // r9
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 *v46; // r12
  __int64 v47; // rdi
  int *v48; // r8
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 *v51; // rax
  unsigned __int64 *v52; // rdx
  __int64 v54; // rdi
  int *v55; // rax
  int *v56; // r8
  __int64 v57; // rsi
  __int64 v58; // rdx
  int *v59; // rax
  int *v60; // r8
  __int64 v61; // rsi
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // rdx
  void (__fastcall *v65)(__m128i *, const __m128i *, __int64); // r9
  _QWORD *v66; // [rsp+8h] [rbp-138h]
  __m128i *v67; // [rsp+10h] [rbp-130h]
  __int64 v68; // [rsp+28h] [rbp-118h] BYREF
  __int64 v69; // [rsp+30h] [rbp-110h] BYREF
  int v70; // [rsp+38h] [rbp-108h] BYREF
  int *v71; // [rsp+40h] [rbp-100h]
  int *v72; // [rsp+48h] [rbp-F8h]
  int *v73; // [rsp+50h] [rbp-F0h]
  __int64 v74; // [rsp+58h] [rbp-E8h]
  __m128i v75; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v76; // [rsp+70h] [rbp-D0h] BYREF
  __m128i v77; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v78; // [rsp+90h] [rbp-B0h] BYREF
  __m128i v79; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i v80; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v81[2]; // [rsp+C0h] [rbp-80h] BYREF
  _QWORD *v82; // [rsp+D0h] [rbp-70h]
  __int64 v83; // [rsp+D8h] [rbp-68h]
  __int64 v84; // [rsp+E0h] [rbp-60h]
  __int64 v85; // [rsp+E8h] [rbp-58h]
  _QWORD *v86; // [rsp+F0h] [rbp-50h]
  __int64 v87; // [rsp+F8h] [rbp-48h]
  __int64 v88; // [rsp+100h] [rbp-40h]
  __int64 *v89; // [rsp+108h] [rbp-38h]

  v70 = 0;
  v71 = 0;
  v72 = &v70;
  v73 = &v70;
  v74 = 0;
  v81[0] = 0;
  v81[1] = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  sub_2BE2EE0(v81, 0);
  v4 = v86;
  v66 = a2 + 1;
  if ( v86 == (_QWORD *)(v88 - 8) )
  {
    sub_2BE2FD0((unsigned __int64 *)v81, v66);
    v5 = (unsigned __int64)v86;
  }
  else
  {
    if ( v86 )
    {
      *v86 = a2[1];
      v4 = v86;
    }
    v5 = (unsigned __int64)(v4 + 1);
    v86 = (_QWORD *)v5;
  }
  if ( v82 != (_QWORD *)v5 )
  {
    do
    {
      if ( v5 == v87 )
      {
        v68 = *(_QWORD *)(*(v89 - 1) + 504);
        j_j___libc_free_0(v5);
        v64 = *--v89 + 512;
        v87 = *v89;
        v88 = v64;
        v6 = v68;
        v86 = (_QWORD *)(v87 + 504);
      }
      else
      {
        v6 = *(_QWORD *)(v5 - 8);
        v86 = (_QWORD *)(v5 - 8);
        v68 = v6;
      }
      v7 = (unsigned __int64 *)*a2;
      v8 = (const __m128i *)(*(_QWORD *)(*a2 + 56LL) + 48 * v6);
      v75 = _mm_loadu_si128(v8);
      v76 = _mm_loadu_si128(v8 + 1);
      v77 = _mm_loadu_si128(v8 + 2);
      if ( v8->m128i_i32[0] == 11 )
      {
        v77.m128i_i64[0] = 0;
        v65 = (void (__fastcall *)(__m128i *, const __m128i *, __int64))v8[2].m128i_i64[0];
        if ( v65 )
        {
          v67 = (__m128i *)v8;
          v65(&v76, v8 + 1, 2);
          v7 = (unsigned __int64 *)*a2;
          v77 = v67[2];
        }
      }
      v9 = _mm_loadu_si128(&v76);
      v10 = _mm_loadu_si128(&v77);
      v78 = _mm_loadu_si128(&v75);
      v79 = v9;
      v80 = v10;
      if ( v75.m128i_i32[0] == 11 )
      {
        v11 = v77.m128i_i64[0];
        v77.m128i_i64[0] = 0;
        v80.m128i_i64[0] = v11;
        v12 = v77.m128i_i64[1];
        v77.m128i_i64[1] = v80.m128i_i64[1];
        v80.m128i_i64[1] = v12;
      }
      v13 = (__m128i *)v7[8];
      if ( v13 == (__m128i *)v7[9] )
      {
        sub_2BE00E0(v7 + 7, v13, &v78);
        v19 = v7[8];
      }
      else
      {
        if ( v13 )
        {
          *v13 = _mm_loadu_si128(&v78);
          v14 = _mm_loadu_si128(&v79);
          v13[1] = v14;
          v13[2] = _mm_loadu_si128(&v80);
          if ( v78.m128i_i32[0] == 11 )
          {
            v13[2].m128i_i64[0] = 0;
            v15 = _mm_loadu_si128(&v79);
            v79 = v14;
            v13[1] = v15;
            v16 = v80.m128i_i64[0];
            v80.m128i_i64[0] = 0;
            v17 = v13[2].m128i_i64[1];
            v13[2].m128i_i64[0] = v16;
            v18 = v80.m128i_i64[1];
            v80.m128i_i64[1] = v17;
            v13[2].m128i_i64[1] = v18;
          }
          v13 = (__m128i *)v7[8];
        }
        v19 = (unsigned __int64)&v13[3];
        v7[8] = v19;
      }
      v20 = v19 - v7[7];
      if ( (unsigned __int64)v20 > 0x493E00 )
        abort();
      v21 = 0xAAAAAAAAAAAAAAABLL * (v20 >> 4) - 1;
      if ( v78.m128i_i32[0] == 11 && v80.m128i_i64[0] )
      {
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v80.m128i_i64[0])(&v79, &v79, 3);
        v22 = v71;
        if ( v71 )
        {
LABEL_20:
          v23 = &v70;
          do
          {
            while ( 1 )
            {
              v24 = *((_QWORD *)v22 + 2);
              v25 = *((_QWORD *)v22 + 3);
              if ( *((_QWORD *)v22 + 4) >= v68 )
                break;
              v22 = (int *)*((_QWORD *)v22 + 3);
              if ( !v25 )
                goto LABEL_24;
            }
            v23 = v22;
            v22 = (int *)*((_QWORD *)v22 + 2);
          }
          while ( v24 );
LABEL_24:
          if ( v23 != &v70 && v68 >= *((_QWORD *)v23 + 4) )
            goto LABEL_27;
          goto LABEL_26;
        }
      }
      else
      {
        v22 = v71;
        if ( v71 )
          goto LABEL_20;
      }
      v23 = &v70;
LABEL_26:
      v78.m128i_i64[0] = (__int64)&v68;
      v23 = (int *)sub_2BE6360(&v69, v23, (__int64 **)&v78);
LABEL_27:
      v26 = v75.m128i_i32[0];
      *((_QWORD *)v23 + 5) = v21;
      if ( ((unsigned int)(v26 - 1) <= 1 || v26 == 7) && v76.m128i_i64[0] != -1 )
      {
        v59 = v71;
        if ( !v71 )
          goto LABEL_92;
        v60 = &v70;
        do
        {
          while ( 1 )
          {
            v61 = *((_QWORD *)v59 + 2);
            v62 = *((_QWORD *)v59 + 3);
            if ( v76.m128i_i64[0] <= *((_QWORD *)v59 + 4) )
              break;
            v59 = (int *)*((_QWORD *)v59 + 3);
            if ( !v62 )
              goto LABEL_90;
          }
          v60 = v59;
          v59 = (int *)*((_QWORD *)v59 + 2);
        }
        while ( v61 );
LABEL_90:
        if ( v60 == &v70 || v76.m128i_i64[0] < *((_QWORD *)v60 + 4) )
        {
LABEL_92:
          v63 = v86;
          if ( v86 == (_QWORD *)(v88 - 8) )
          {
            sub_2BE2FD0((unsigned __int64 *)v81, &v76);
          }
          else
          {
            if ( v86 )
            {
              *v86 = v76.m128i_i64[0];
              v63 = v86;
            }
            v86 = v63 + 1;
          }
        }
      }
      if ( a2[2] != v68 && v75.m128i_i64[1] != -1 )
      {
        v27 = v71;
        if ( !v71 )
          goto LABEL_38;
        v28 = &v70;
        do
        {
          while ( 1 )
          {
            v29 = *((_QWORD *)v27 + 2);
            v30 = *((_QWORD *)v27 + 3);
            if ( v75.m128i_i64[1] <= *((_QWORD *)v27 + 4) )
              break;
            v27 = (int *)*((_QWORD *)v27 + 3);
            if ( !v30 )
              goto LABEL_36;
          }
          v28 = v27;
          v27 = (int *)*((_QWORD *)v27 + 2);
        }
        while ( v29 );
LABEL_36:
        if ( v28 == &v70 || v75.m128i_i64[1] < *((_QWORD *)v28 + 4) )
        {
LABEL_38:
          v31 = v86;
          if ( v86 == (_QWORD *)(v88 - 8) )
          {
            sub_2BE2FD0((unsigned __int64 *)v81, &v75.m128i_i64[1]);
          }
          else
          {
            if ( v86 )
            {
              *v86 = v75.m128i_i64[1];
              v31 = v86;
            }
            v86 = v31 + 1;
          }
        }
      }
      if ( v75.m128i_i32[0] == 11 && v77.m128i_i64[0] )
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v77.m128i_i64[0])(&v76, &v76, 3);
      v5 = (unsigned __int64)v86;
    }
    while ( v82 != v86 );
  }
  for ( i = (__int64)v72; (int *)i != &v70; i = sub_220EEE0(i) )
  {
    v33 = *(_QWORD *)(*(_QWORD *)(*a2 + 56LL) + 48LL * *(_QWORD *)(i + 40) + 8);
    v34 = *(_QWORD *)(*a2 + 56LL) + 48LL * *(_QWORD *)(i + 40);
    if ( v33 != -1 )
    {
      v35 = v71;
      v36 = &v70;
      if ( v71 )
      {
        do
        {
          while ( 1 )
          {
            v37 = *((_QWORD *)v35 + 2);
            v38 = *((_QWORD *)v35 + 3);
            if ( v33 <= *((_QWORD *)v35 + 4) )
              break;
            v35 = (int *)*((_QWORD *)v35 + 3);
            if ( !v38 )
              goto LABEL_51;
          }
          v36 = v35;
          v35 = (int *)*((_QWORD *)v35 + 2);
        }
        while ( v37 );
LABEL_51:
        if ( v36 != &v70 && v33 < *((_QWORD *)v36 + 4) )
          v36 = &v70;
      }
      *(_QWORD *)(v34 + 8) = *((_QWORD *)v36 + 5);
    }
    if ( (unsigned int)(*(_DWORD *)v34 - 1) <= 1 || *(_DWORD *)v34 == 7 )
    {
      v54 = *(_QWORD *)(v34 + 16);
      if ( v54 != -1 )
      {
        v55 = v71;
        v56 = &v70;
        if ( v71 )
        {
          do
          {
            while ( 1 )
            {
              v57 = *((_QWORD *)v55 + 2);
              v58 = *((_QWORD *)v55 + 3);
              if ( v54 <= *((_QWORD *)v55 + 4) )
                break;
              v55 = (int *)*((_QWORD *)v55 + 3);
              if ( !v58 )
                goto LABEL_80;
            }
            v56 = v55;
            v55 = (int *)*((_QWORD *)v55 + 2);
          }
          while ( v57 );
LABEL_80:
          if ( v56 != &v70 && v54 < *((_QWORD *)v56 + 4) )
            v56 = &v70;
        }
        *(_QWORD *)(v34 + 16) = *((_QWORD *)v56 + 5);
      }
    }
  }
  v39 = v71;
  if ( v71 )
  {
    v40 = a2[2];
    v41 = v71;
    v42 = &v70;
    do
    {
      while ( 1 )
      {
        v43 = *((_QWORD *)v41 + 2);
        v44 = *((_QWORD *)v41 + 3);
        if ( *((_QWORD *)v41 + 4) >= v40 )
          break;
        v41 = (int *)*((_QWORD *)v41 + 3);
        if ( !v44 )
          goto LABEL_63;
      }
      v42 = v41;
      v41 = (int *)*((_QWORD *)v41 + 2);
    }
    while ( v43 );
LABEL_63:
    if ( v42 != &v70 && v40 >= *((_QWORD *)v42 + 4) )
    {
      v46 = (unsigned __int64 *)*((_QWORD *)v42 + 5);
      goto LABEL_66;
    }
  }
  else
  {
    v42 = &v70;
  }
  v78.m128i_i64[0] = (__int64)(a2 + 2);
  v45 = sub_2BE6360(&v69, v42, (__int64 **)&v78);
  v39 = v71;
  v46 = *(unsigned __int64 **)(v45 + 40);
  if ( !v71 )
  {
    v48 = &v70;
LABEL_72:
    v78.m128i_i64[0] = (__int64)v66;
    v48 = (int *)sub_2BE6360(&v69, v48, (__int64 **)&v78);
    goto LABEL_73;
  }
LABEL_66:
  v47 = a2[1];
  v48 = &v70;
  do
  {
    while ( 1 )
    {
      v49 = *((_QWORD *)v39 + 2);
      v50 = *((_QWORD *)v39 + 3);
      if ( *((_QWORD *)v39 + 4) >= v47 )
        break;
      v39 = (int *)*((_QWORD *)v39 + 3);
      if ( !v50 )
        goto LABEL_70;
    }
    v48 = v39;
    v39 = (int *)*((_QWORD *)v39 + 2);
  }
  while ( v49 );
LABEL_70:
  if ( v48 == &v70 || v47 < *((_QWORD *)v48 + 4) )
    goto LABEL_72;
LABEL_73:
  v51 = (unsigned __int64 *)*((_QWORD *)v48 + 5);
  v52 = (unsigned __int64 *)*a2;
  a1[2] = v46;
  a1[1] = v51;
  *a1 = v52;
  sub_2BE1050((unsigned __int64 *)v81);
  sub_2BDC3F0((unsigned __int64)v71);
  return a1;
}
