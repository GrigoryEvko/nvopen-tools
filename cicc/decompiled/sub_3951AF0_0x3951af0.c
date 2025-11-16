// Function: sub_3951AF0
// Address: 0x3951af0
//
__int64 __fastcall sub_3951AF0(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 *v2; // rax
  __int64 v3; // rbx
  __m128i v4; // xmm3
  __int64 *v5; // r12
  __int64 **v6; // r9
  int v7; // r10d
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  unsigned int i; // edx
  __int64 **v12; // rax
  __int64 *v13; // rcx
  unsigned __int64 v14; // rdi
  __int64 **v16; // rdi
  int v17; // r8d
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  unsigned int v20; // esi
  __int64 *v21; // r9
  unsigned int v22; // esi
  unsigned int v23; // edx
  int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // r13
  __m128i v29; // xmm1
  __int64 v30; // r14
  __int64 v31; // r12
  int v32; // r8d
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  unsigned int k; // eax
  _QWORD *v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rax
  int v39; // ebx
  unsigned int v40; // r9d
  unsigned __int64 v41; // r8
  unsigned int v42; // edx
  __int64 v43; // rdi
  __int64 m; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // rax
  size_t v48; // rdx
  unsigned int v49; // r9d
  void *v50; // r8
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  int v53; // ebx
  int v54; // edi
  unsigned int j; // ebx
  __int64 **v56; // rdx
  __int64 *v57; // r8
  unsigned int v58; // ebx
  __int64 v59; // rax
  void *v60; // [rsp+8h] [rbp-188h]
  __int64 *v61; // [rsp+18h] [rbp-178h]
  unsigned __int8 v62; // [rsp+27h] [rbp-169h]
  __int64 *v63; // [rsp+28h] [rbp-168h]
  __int64 v64; // [rsp+38h] [rbp-158h]
  __int64 *v65; // [rsp+40h] [rbp-150h]
  __int64 v67; // [rsp+50h] [rbp-140h]
  __int64 *v68; // [rsp+58h] [rbp-138h]
  __int64 v69; // [rsp+60h] [rbp-130h]
  size_t v70; // [rsp+68h] [rbp-128h]
  unsigned int n; // [rsp+70h] [rbp-120h]
  unsigned int na; // [rsp+70h] [rbp-120h]
  __int64 v73; // [rsp+78h] [rbp-118h]
  __int64 v74; // [rsp+80h] [rbp-110h] BYREF
  unsigned __int64 v75; // [rsp+88h] [rbp-108h]
  __int64 v76; // [rsp+90h] [rbp-100h]
  __int64 v77; // [rsp+98h] [rbp-F8h]
  __m128i v78; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v79; // [rsp+B0h] [rbp-E0h]
  __m128i v80; // [rsp+C0h] [rbp-D0h] BYREF
  __m128i v81; // [rsp+D0h] [rbp-C0h]
  __m128i v82; // [rsp+E0h] [rbp-B0h] BYREF
  __m128i v83; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v84; // [rsp+100h] [rbp-90h]
  __m128i v85; // [rsp+120h] [rbp-70h] BYREF
  __m128i v86; // [rsp+130h] [rbp-60h] BYREF
  __int64 v87; // [rsp+140h] [rbp-50h]

  v1 = a1[6];
  v2 = (__int64 *)a1[5];
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v61 = (__int64 *)v1;
  if ( v2 == (__int64 *)v1 )
  {
    v62 = 0;
    v14 = 0;
    goto LABEL_18;
  }
  v63 = v2;
  v62 = 0;
  do
  {
    v3 = *v63;
    v65 = (__int64 *)*v63;
    sub_3950AD0(v82.m128i_i64, *v63, *a1);
    v4 = _mm_loadu_si128(&v83);
    v78 = _mm_loadu_si128(&v82);
    v67 = v84;
    v79 = v4;
    if ( v84 == v82.m128i_i64[0] )
      goto LABEL_16;
    v64 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
    do
    {
      v5 = (__int64 *)v79.m128i_i64[0];
      if ( !(_DWORD)v77 )
      {
        ++v74;
        goto LABEL_20;
      }
      v6 = 0;
      v7 = 1;
      v8 = (((v64
            | ((unsigned __int64)(((unsigned __int32)v79.m128i_i32[0] >> 9) ^ ((unsigned __int32)v79.m128i_i32[0] >> 4)) << 32))
           - 1
           - (v64 << 32)) >> 22)
         ^ ((v64
           | ((unsigned __int64)(((unsigned __int32)v79.m128i_i32[0] >> 9) ^ ((unsigned __int32)v79.m128i_i32[0] >> 4)) << 32))
          - 1
          - (v64 << 32));
      v9 = ((v8 - 1 - (v8 << 13)) >> 8) ^ (v8 - 1 - (v8 << 13));
      v10 = (((((9 * v9) >> 15) ^ (9 * v9)) - 1 - ((((9 * v9) >> 15) ^ (9 * v9)) << 27)) >> 31)
          ^ ((((9 * v9) >> 15) ^ (9 * v9)) - 1 - ((((9 * v9) >> 15) ^ (9 * v9)) << 27));
      for ( i = v10 & (v77 - 1); ; i = (v77 - 1) & v23 )
      {
        v12 = (__int64 **)(v75 + 16LL * i);
        v13 = *v12;
        if ( (__int64 *)v79.m128i_i64[0] == *v12 && v65 == v12[1] )
          goto LABEL_15;
        if ( v13 == (__int64 *)-8LL )
          break;
        if ( v13 == (__int64 *)-16LL && v12[1] == (__int64 *)-16LL && !v6 )
          v6 = (__int64 **)(v75 + 16LL * i);
LABEL_30:
        v23 = v7 + i;
        ++v7;
      }
      if ( v12[1] != (__int64 *)-8LL )
        goto LABEL_30;
      if ( v6 )
        v12 = v6;
      ++v74;
      v24 = v76 + 1;
      if ( 4 * ((int)v76 + 1) < (unsigned int)(3 * v77) )
      {
        if ( (int)v77 - HIDWORD(v76) - v24 > (unsigned int)v77 >> 3 )
          goto LABEL_35;
        sub_3951850((__int64)&v74, v77);
        if ( (_DWORD)v77 )
        {
          v54 = 1;
          v12 = 0;
          for ( j = (v77 - 1) & v10; ; j = (v77 - 1) & v58 )
          {
            v56 = (__int64 **)(v75 + 16LL * j);
            v57 = *v56;
            if ( v5 == *v56 && v65 == v56[1] )
            {
              v24 = v76 + 1;
              v12 = (__int64 **)(v75 + 16LL * j);
              goto LABEL_35;
            }
            if ( v57 == (__int64 *)-8LL )
            {
              if ( v56[1] == (__int64 *)-8LL )
              {
                if ( !v12 )
                  v12 = (__int64 **)(v75 + 16LL * j);
                v24 = v76 + 1;
                goto LABEL_35;
              }
            }
            else if ( v57 == (__int64 *)-16LL && v56[1] == (__int64 *)-16LL && !v12 )
            {
              v12 = (__int64 **)(v75 + 16LL * j);
            }
            v58 = v54 + j;
            ++v54;
          }
        }
LABEL_98:
        LODWORD(v76) = v76 + 1;
        BUG();
      }
LABEL_20:
      sub_3951850((__int64)&v74, 2 * v77);
      if ( !(_DWORD)v77 )
        goto LABEL_98;
      v16 = 0;
      v17 = 1;
      v18 = (((v64 | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32)) - 1 - (v64 << 32)) >> 22)
          ^ ((v64 | ((unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32)) - 1 - (v64 << 32));
      v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
          ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
      v20 = (v77 - 1) & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27)));
      while ( 2 )
      {
        v12 = (__int64 **)(v75 + 16LL * v20);
        v21 = *v12;
        if ( v5 == *v12 && v65 == v12[1] )
        {
          v24 = v76 + 1;
          goto LABEL_35;
        }
        if ( v21 != (__int64 *)-8LL )
        {
          if ( v21 == (__int64 *)-16LL && v12[1] == (__int64 *)-16LL && !v16 )
            v16 = (__int64 **)(v75 + 16LL * v20);
          goto LABEL_28;
        }
        if ( v12[1] != (__int64 *)-8LL )
        {
LABEL_28:
          v22 = v17 + v20;
          ++v17;
          v20 = (v77 - 1) & v22;
          continue;
        }
        break;
      }
      if ( v16 )
        v12 = v16;
      v24 = v76 + 1;
LABEL_35:
      LODWORD(v76) = v24;
      if ( *v12 != (__int64 *)-8LL || v12[1] != (__int64 *)-8LL )
        --HIDWORD(v76);
      *v12 = v5;
      v12[1] = v65;
      v25 = sub_3950CA0(v5, v65, (__int64)a1);
      v26 = *a1;
      v27 = (__int64)v25;
      v68 = v25;
      v28 = sub_3950BA0((__int64)a1, v25);
      sub_3950AD0(v85.m128i_i64, v27, v26);
      v29 = _mm_loadu_si128(&v86);
      v30 = v87;
      v80 = _mm_loadu_si128(&v85);
      v81 = v29;
      if ( v85.m128i_i64[0] == v87 )
        goto LABEL_15;
      v31 = ((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4);
      while ( 2 )
      {
        if ( !(_DWORD)v77 )
          goto LABEL_45;
        v32 = 1;
        v33 = (((v31
               | ((unsigned __int64)(((unsigned __int32)v81.m128i_i32[0] >> 9)
                                   ^ ((unsigned __int32)v81.m128i_i32[0] >> 4)) << 32))
              - 1
              - (v31 << 32)) >> 22)
            ^ ((v31
              | ((unsigned __int64)(((unsigned __int32)v81.m128i_i32[0] >> 9) ^ ((unsigned __int32)v81.m128i_i32[0] >> 4)) << 32))
             - 1
             - (v31 << 32));
        v34 = ((9 * (((v33 - 1 - (v33 << 13)) >> 8) ^ (v33 - 1 - (v33 << 13)))) >> 15)
            ^ (9 * (((v33 - 1 - (v33 << 13)) >> 8) ^ (v33 - 1 - (v33 << 13))));
        for ( k = (v77 - 1) & (((v34 - 1 - (v34 << 27)) >> 31) ^ (v34 - 1 - ((_DWORD)v34 << 27))); ; k = (v77 - 1) & v37 )
        {
          v36 = (_QWORD *)(v75 + 16LL * k);
          if ( v81.m128i_i64[0] == *v36 && v68 == (__int64 *)v36[1] )
            break;
          if ( *v36 == -8 && v36[1] == -8 )
            goto LABEL_45;
          v37 = v32 + k;
          ++v32;
        }
        v38 = sub_3950BA0((__int64)a1, (__int64 *)v81.m128i_i64[0]);
        v39 = *(_DWORD *)(v38 + 24);
        v69 = v38;
        if ( v39 )
        {
          v73 = (unsigned int)(v39 + 63) >> 6;
          v47 = malloc(8 * v73);
          v48 = 8 * v73;
          v49 = (unsigned int)(v39 + 63) >> 6;
          v50 = (void *)v47;
          if ( !v47 )
          {
            if ( 8 * v73 || (v59 = malloc(1u), v49 = (unsigned int)(v39 + 63) >> 6, v48 = 0, v50 = 0, !v59) )
            {
              v60 = v50;
              v70 = v48;
              na = v49;
              sub_16BD1C0("Allocation failed", 1u);
              v49 = na;
              v48 = v70;
              v50 = v60;
            }
            else
            {
              v50 = (void *)v59;
            }
          }
          n = v49;
          v51 = memcpy(v50, *(const void **)(v69 + 8), v48);
          v40 = n;
          v41 = (unsigned __int64)v51;
          if ( n )
          {
            v52 = &v51[v73];
            do
            {
              *v51 = ~*v51;
              ++v51;
            }
            while ( v52 != v51 );
          }
          v53 = v39 & 0x3F;
          if ( v53 )
            *(_QWORD *)(v41 + 8LL * (n - 1)) &= ~(-1LL << v53);
        }
        else
        {
          v40 = 0;
          v41 = 0;
        }
        v42 = (unsigned int)(*(_DWORD *)(v28 + 24) + 63) >> 6;
        if ( v42 > v40 )
          v42 = v40;
        if ( v42 )
        {
          v43 = *(_QWORD *)(v28 + 8);
          for ( m = 0; m != v42; ++m )
            *(_QWORD *)(v41 + 8 * m) &= *(_QWORD *)(v43 + 8 * m);
        }
        for ( ; v40 != v42; *(_QWORD *)(v41 + 8 * v45) = 0 )
          v45 = v42++;
        if ( !v40 )
        {
LABEL_44:
          _libc_free(v41);
LABEL_45:
          v80.m128i_i64[0] = *(_QWORD *)(v80.m128i_i64[0] + 8);
          sub_3950920(v80.m128i_i64);
          if ( v30 == v80.m128i_i64[0] )
            goto LABEL_15;
          continue;
        }
        break;
      }
      v46 = (_QWORD *)v41;
      while ( !*v46 )
      {
        if ( (_QWORD *)(v41 + 8LL * (v40 - 1) + 8) == ++v46 )
          goto LABEL_44;
      }
      _libc_free(v41);
      v62 = 1;
LABEL_15:
      v78.m128i_i64[0] = *(_QWORD *)(v78.m128i_i64[0] + 8);
      sub_3950920(v78.m128i_i64);
    }
    while ( v78.m128i_i64[0] != v67 );
LABEL_16:
    ++v63;
  }
  while ( v61 != v63 );
  v14 = v75;
LABEL_18:
  j___libc_free_0(v14);
  return v62;
}
