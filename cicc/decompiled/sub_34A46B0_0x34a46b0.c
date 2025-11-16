// Function: sub_34A46B0
// Address: 0x34a46b0
//
void __fastcall sub_34A46B0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, unsigned __int64 a6)
{
  char v6; // al
  __int64 *v7; // r15
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // r14
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // r13
  __int64 v16; // rax
  int v17; // edi
  int *v18; // rsi
  int v19; // edx
  __int64 v20; // rdx
  __int64 v21; // r15
  unsigned __int64 v22; // r14
  int v23; // eax
  __m128i *v24; // rdi
  __int64 v25; // rdx
  int v26; // r9d
  __int64 v27; // rdx
  __int64 v28; // rbx
  unsigned __int64 v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rbx
  unsigned __int64 v35; // r12
  const __m128i *v36; // rbx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rdi
  unsigned int v39; // esi
  unsigned __int64 v40; // r12
  const __m128i *v41; // r13
  const __m128i *v42; // r12
  __m128i *v43; // rbx
  __m128i *v44; // rdi
  __m128i *v45; // rdi
  __int32 v46; // eax
  __int64 v47; // r8
  __int64 v48; // rax
  const __m128i *v49; // rsi
  size_t v50; // rdx
  __int64 v51; // rax
  const __m128i *v52; // rsi
  size_t v53; // rdx
  const __m128i *v54; // rbx
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdi
  int v57; // ebx
  unsigned __int64 v59; // [rsp+20h] [rbp-3280h]
  __int64 v60; // [rsp+28h] [rbp-3278h]
  int v61; // [rsp+30h] [rbp-3270h]
  unsigned int v62; // [rsp+30h] [rbp-3270h]
  unsigned int v63; // [rsp+30h] [rbp-3270h]
  __int32 v64; // [rsp+30h] [rbp-3270h]
  char v65; // [rsp+37h] [rbp-3269h]
  __int64 *v66; // [rsp+40h] [rbp-3260h]
  __int64 *v67; // [rsp+48h] [rbp-3258h]
  __int64 v68; // [rsp+58h] [rbp-3248h]
  unsigned __int64 v69; // [rsp+68h] [rbp-3238h] BYREF
  __int64 v70; // [rsp+70h] [rbp-3230h] BYREF
  unsigned __int64 v71; // [rsp+78h] [rbp-3228h] BYREF
  unsigned int v72; // [rsp+80h] [rbp-3220h]
  char v73; // [rsp+88h] [rbp-3218h] BYREF
  unsigned int v74; // [rsp+C8h] [rbp-31D8h]
  unsigned __int64 v75; // [rsp+D0h] [rbp-31D0h]
  unsigned __int64 v76; // [rsp+D8h] [rbp-31C8h]
  __m128i v77; // [rsp+E0h] [rbp-31C0h] BYREF
  __int64 v78; // [rsp+F0h] [rbp-31B0h]
  char v79; // [rsp+F8h] [rbp-31A8h] BYREF
  int v80; // [rsp+118h] [rbp-3188h]
  char *v81; // [rsp+120h] [rbp-3180h]
  char v82; // [rsp+130h] [rbp-3170h] BYREF
  int v83; // [rsp+138h] [rbp-3168h]
  __int64 v84; // [rsp+140h] [rbp-3160h]
  __int64 v85; // [rsp+148h] [rbp-3158h]
  _BYTE *v86; // [rsp+230h] [rbp-3070h]
  _BYTE v87[32]; // [rsp+240h] [rbp-3060h] BYREF
  const __m128i *v88; // [rsp+260h] [rbp-3040h] BYREF
  __int64 v89; // [rsp+268h] [rbp-3038h]
  _BYTE v90[12336]; // [rsp+270h] [rbp-3030h] BYREF

  v6 = *(_BYTE *)(a1 + 8) & 1;
  if ( *(_DWORD *)(a1 + 8) >> 1 )
  {
    if ( v6 )
    {
      v9 = (__int64 *)(a1 + 80);
      v7 = (__int64 *)(a1 + 16);
      v66 = (__int64 *)(a1 + 80);
      goto LABEL_4;
    }
    v7 = *(__int64 **)(a1 + 16);
    v8 = 2LL * *(unsigned int *)(a1 + 24);
    v9 = &v7[v8];
    v66 = &v7[v8];
    if ( v7 != &v7[v8] )
    {
LABEL_4:
      while ( *v7 == -8192 || *v7 == -4096 )
      {
        v7 += 2;
        if ( v9 == v7 )
          return;
      }
    }
  }
  else
  {
    if ( v6 )
    {
      v10 = a1 + 16;
      v11 = 64;
    }
    else
    {
      v10 = *(_QWORD *)(a1 + 16);
      v11 = 16LL * *(unsigned int *)(a1 + 24);
    }
    v7 = (__int64 *)(v10 + v11);
    v66 = (__int64 *)(v10 + v11);
  }
  if ( v66 == v7 )
    return;
  v12 = v7;
  do
  {
    v13 = v12[1];
    v14 = *v12;
    v88 = (const __m128i *)v90;
    v89 = 0x2000000000LL;
    sub_34A3D10((__int64)&v70, v13, 0, a4, a5, a6);
    v68 = v14;
    v77.m128i_i64[0] = 0;
    a4 = v75;
    v77.m128i_i64[1] = (__int64)&v79;
    v15 = a2;
    v78 = 0x400000000LL;
    v16 = v74;
    v84 = 0;
    v85 = 0;
    v83 = -1;
    v67 = v12;
LABEL_21:
    if ( (_DWORD)v16 != -1 )
      goto LABEL_22;
    while ( v76 | a4 )
    {
      a4 += 0xFFFFFFFFLL;
      if ( a4 > 0xFFFFFFFF )
        break;
LABEL_23:
      if ( (*(_BYTE *)(v15 + 56) & 1) != 0 )
      {
        v17 = *(_DWORD *)(v15 + 64);
        v18 = (int *)(v15 + 64);
        v19 = 3;
        if ( !v17 )
          goto LABEL_16;
LABEL_26:
        v26 = 1;
        a5 = 0;
        while ( v17 != -1 )
        {
          a5 = v19 & (unsigned int)(v26 + a5);
          v17 = v18[8 * (unsigned int)a5];
          if ( !v17 )
          {
            v18 += 8 * (unsigned int)a5;
            goto LABEL_16;
          }
          ++v26;
        }
        if ( (*(_BYTE *)(v15 + 56) & 1) != 0 )
        {
          v27 = 128;
          goto LABEL_65;
        }
        v25 = *(unsigned int *)(v15 + 72);
        goto LABEL_64;
      }
      v25 = *(unsigned int *)(v15 + 72);
      v18 = *(int **)(v15 + 64);
      if ( (_DWORD)v25 )
      {
        v17 = *v18;
        v19 = v25 - 1;
        if ( !*v18 )
          goto LABEL_16;
        goto LABEL_26;
      }
LABEL_64:
      v27 = 32 * v25;
LABEL_65:
      v18 = (int *)((char *)v18 + v27);
LABEL_16:
      v20 = (unsigned int)v89;
      v21 = (__int64)v88;
      v22 = *((_QWORD *)v18 + 1) + 384 * a4;
      a6 = (unsigned int)v89 + 1LL;
      v23 = v89;
      if ( a6 > HIDWORD(v89) )
      {
        if ( (unsigned __int64)v88 > v22 || v22 >= (unsigned __int64)&v88[24 * (unsigned int)v89] )
        {
          v59 = -1;
          v65 = 0;
        }
        else
        {
          v65 = 1;
          v59 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v22 - (_QWORD)v88) >> 7);
        }
        v21 = sub_C8D7D0((__int64)&v88, (__int64)v90, (unsigned int)v89 + 1LL, 0x180u, &v69, a6);
        v40 = (unsigned __int64)&v88[24 * (unsigned int)v89];
        if ( v88 != (const __m128i *)v40 )
        {
          v60 = v15;
          v41 = &v88[24 * (unsigned int)v89];
          v42 = v88 + 22;
          v43 = (__m128i *)v21;
          while ( 1 )
          {
            if ( v43 )
            {
              v45 = v43 + 5;
              *v43 = _mm_loadu_si128(v42 - 22);
              v43[1] = _mm_loadu_si128(v42 - 21);
              v43[2].m128i_i64[0] = v42[-20].m128i_i64[0];
              v43[2].m128i_i64[1] = v42[-20].m128i_i64[1];
              v43[3].m128i_i64[0] = v42[-19].m128i_i64[0];
              v46 = v42[-19].m128i_i32[2];
              v43[4].m128i_i64[0] = (__int64)v43[5].m128i_i64;
              v43[3].m128i_i32[2] = v46;
              v43[4].m128i_i32[2] = 0;
              v43[4].m128i_i32[3] = 8;
              v47 = v42[-18].m128i_u32[2];
              if ( (_DWORD)v47 && &v43[4] != &v42[-18] )
              {
                v48 = v42[-18].m128i_i64[0];
                v49 = v42 - 17;
                if ( (const __m128i *)v48 == &v42[-17] )
                {
                  v50 = 32LL * (unsigned int)v47;
                  if ( (unsigned int)v47 <= 8
                    || (v64 = v42[-18].m128i_i32[2],
                        sub_C8D5F0((__int64)v43[4].m128i_i64, &v43[5], (unsigned int)v47, 0x20u, v47, a6),
                        v45 = (__m128i *)v43[4].m128i_i64[0],
                        v49 = (const __m128i *)v42[-18].m128i_i64[0],
                        LODWORD(v47) = v64,
                        (v50 = 32LL * v42[-18].m128i_u32[2]) != 0) )
                  {
                    v61 = v47;
                    memcpy(v45, v49, v50);
                    LODWORD(v47) = v61;
                  }
                  v43[4].m128i_i32[2] = v47;
                  v42[-18].m128i_i32[2] = 0;
                }
                else
                {
                  v43[4].m128i_i64[0] = v48;
                  v43[4].m128i_i32[2] = v42[-18].m128i_i32[2];
                  v43[4].m128i_i32[3] = v42[-18].m128i_i32[3];
                  v42[-18].m128i_i64[0] = (__int64)v49;
                  v42[-18].m128i_i32[3] = 0;
                  v42[-18].m128i_i32[2] = 0;
                }
              }
              v44 = v43 + 22;
              v43[21].m128i_i32[2] = 0;
              v43[21].m128i_i64[0] = (__int64)v43[22].m128i_i64;
              v43[21].m128i_i32[3] = 8;
              a5 = v42[-1].m128i_u32[2];
              if ( (_DWORD)a5 && &v43[21] != &v42[-1] )
              {
                v51 = v42[-1].m128i_i64[0];
                if ( (const __m128i *)v51 == v42 )
                {
                  v52 = v42;
                  v53 = 4LL * (unsigned int)a5;
                  if ( (unsigned int)a5 <= 8
                    || (v63 = v42[-1].m128i_u32[2],
                        sub_C8D5F0((__int64)v43[21].m128i_i64, &v43[22], (unsigned int)a5, 4u, a5, a6),
                        v44 = (__m128i *)v43[21].m128i_i64[0],
                        v52 = (const __m128i *)v42[-1].m128i_i64[0],
                        a5 = v63,
                        (v53 = 4LL * v42[-1].m128i_u32[2]) != 0) )
                  {
                    v62 = a5;
                    memcpy(v44, v52, v53);
                    a5 = v62;
                  }
                  v43[21].m128i_i32[2] = a5;
                  v42[-1].m128i_i32[2] = 0;
                }
                else
                {
                  v43[21].m128i_i64[0] = v51;
                  v43[21].m128i_i32[2] = v42[-1].m128i_i32[2];
                  v43[21].m128i_i32[3] = v42[-1].m128i_i32[3];
                  v42[-1].m128i_i64[0] = (__int64)v42;
                  v42[-1].m128i_i32[3] = 0;
                  v42[-1].m128i_i32[2] = 0;
                }
              }
            }
            v43 += 24;
            if ( v41 == &v42[2] )
              break;
            v42 += 24;
          }
          v54 = v88;
          v15 = v60;
          v40 = (unsigned __int64)&v88[24 * (unsigned int)v89];
          if ( v88 != (const __m128i *)v40 )
          {
            do
            {
              v40 -= 384LL;
              v55 = *(_QWORD *)(v40 + 336);
              if ( v55 != v40 + 352 )
                _libc_free(v55);
              v56 = *(_QWORD *)(v40 + 64);
              if ( v56 != v40 + 80 )
                _libc_free(v56);
            }
            while ( (const __m128i *)v40 != v54 );
            v40 = (unsigned __int64)v88;
          }
        }
        v57 = v69;
        if ( (_BYTE *)v40 != v90 )
          _libc_free(v40);
        v20 = (unsigned int)v89;
        v88 = (const __m128i *)v21;
        HIDWORD(v89) = v57;
        v23 = v89;
        if ( v65 )
          v22 = v21 + 384 * v59;
      }
      v24 = (__m128i *)(v21 + 384 * v20);
      if ( v24 )
      {
        sub_349DE60(v24, v22);
        v23 = v89;
      }
      a4 = v75;
      LODWORD(v89) = v23 + 1;
      if ( v75 + v74 < v76 )
      {
        v16 = ++v74;
        goto LABEL_21;
      }
      v32 = v71 + 16LL * v72 - 16;
      v33 = *(_DWORD *)(v32 + 12) + 1;
      *(_DWORD *)(v32 + 12) = v33;
      v34 = v72;
      if ( v33 == *(_DWORD *)(v71 + 16LL * v72 - 8) )
      {
        v39 = *(_DWORD *)(v70 + 192);
        if ( v39 )
        {
          sub_F03D40((__int64 *)&v71, v39);
          v34 = v72;
        }
      }
      if ( (_DWORD)v34 )
      {
        v35 = v71;
        if ( *(_DWORD *)(v71 + 12) < *(_DWORD *)(v71 + 8) )
        {
          v74 = 0;
          a4 = *(_QWORD *)sub_34A2590((__int64)&v70);
          v75 = a4;
          v76 = *(_QWORD *)(*(_QWORD *)(v35 + 16 * v34 - 16) + 16LL * *(unsigned int *)(v35 + 16 * v34 - 16 + 12) + 8);
          v16 = 0;
LABEL_22:
          a4 += v16;
          if ( a4 > 0xFFFFFFFF )
            break;
          goto LABEL_23;
        }
      }
      v74 = -1;
      a4 = 0;
      v75 = 0;
      v76 = 0;
    }
    if ( (char *)v71 != &v73 )
      _libc_free(v71);
    v28 = (__int64)v88;
    v29 = (unsigned __int64)&v88[24 * (unsigned int)v89];
    if ( v88 == (const __m128i *)v29 )
      goto LABEL_56;
    while ( 2 )
    {
      sub_349DE60(&v77, v28);
      if ( (unsigned int)(v80 - 2) > 1 )
      {
        v30 = sub_349FA50((__int64)&v77, *(_QWORD *)(v68 + 32));
        sub_2E326B0(v68, *(__int64 **)(v68 + 56), v30);
        v31 = (unsigned __int64)v86;
        if ( v86 == v87 )
          goto LABEL_39;
LABEL_38:
        _libc_free(v31);
        goto LABEL_39;
      }
      v31 = (unsigned __int64)v86;
      if ( v86 != v87 )
        goto LABEL_38;
LABEL_39:
      if ( v81 != &v82 )
        _libc_free((unsigned __int64)v81);
      v28 += 384;
      if ( v29 != v28 )
        continue;
      break;
    }
    v36 = v88;
    v29 = (unsigned __int64)&v88[24 * (unsigned int)v89];
    if ( v88 != (const __m128i *)v29 )
    {
      do
      {
        v29 -= 384LL;
        v37 = *(_QWORD *)(v29 + 336);
        if ( v37 != v29 + 352 )
          _libc_free(v37);
        v38 = *(_QWORD *)(v29 + 64);
        if ( v38 != v29 + 80 )
          _libc_free(v38);
      }
      while ( v36 != (const __m128i *)v29 );
      v29 = (unsigned __int64)v88;
    }
LABEL_56:
    if ( (_BYTE *)v29 != v90 )
      _libc_free(v29);
    v12 = v67 + 2;
    if ( v67 + 2 == v66 )
      break;
    while ( *v12 == -8192 || *v12 == -4096 )
    {
      v12 += 2;
      if ( v66 == v12 )
        return;
    }
  }
  while ( v12 != v66 );
}
