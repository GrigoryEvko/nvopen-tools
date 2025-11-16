// Function: sub_120D220
// Address: 0x120d220
//
__int64 __fastcall sub_120D220(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r13
  int v4; // eax
  unsigned __int64 v5; // rsi
  unsigned int v6; // r15d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __m128i v19; // xmm0
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  const char **v22; // r10
  char v23; // dl
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  char *v27; // r11
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  __m128i v30; // xmm0
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __m128i v33; // xmm3
  unsigned __int64 v34; // rax
  __int64 i; // rdx
  __int64 v36; // r12
  unsigned __int64 v37; // r12
  __int64 v38; // rsi
  __int64 v39; // r12
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // [rsp+18h] [rbp-188h]
  __m128i v43; // [rsp+30h] [rbp-170h] BYREF
  _QWORD v44[2]; // [rsp+40h] [rbp-160h] BYREF
  _QWORD v45[4]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v46; // [rsp+70h] [rbp-130h]
  char v47; // [rsp+80h] [rbp-120h] BYREF
  __m128i v48; // [rsp+88h] [rbp-118h] BYREF
  __m128i v49; // [rsp+98h] [rbp-108h] BYREF
  __m128i v50; // [rsp+A8h] [rbp-F8h] BYREF
  char v51; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v52; // [rsp+C8h] [rbp-D8h] BYREF
  __m128i v53; // [rsp+D8h] [rbp-C8h] BYREF
  __m128i v54; // [rsp+E8h] [rbp-B8h] BYREF
  const char *v55; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v56; // [rsp+108h] [rbp-98h] BYREF
  __m128i v57; // [rsp+118h] [rbp-88h] BYREF
  __m128i v58; // [rsp+128h] [rbp-78h] BYREF
  char v59; // [rsp+138h] [rbp-68h] BYREF
  __m128i v60; // [rsp+140h] [rbp-60h] BYREF
  __m128i v61; // [rsp+150h] [rbp-50h] BYREF
  __m128i v62; // [rsp+160h] [rbp-40h] BYREF

  v2 = a1 + 176;
  v4 = sub_1205200(a1 + 176);
  v5 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = v4;
  if ( v4 == 12 )
  {
    v43.m128i_i64[1] = 0;
    *(_DWORD *)(a1 + 240) = sub_1205200(v2);
    v8 = *(_QWORD *)(a1 + 232);
    LOBYTE(v44[0]) = 0;
    v41 = v8;
    v43.m128i_i64[0] = (__int64)v44;
    v6 = sub_120B3D0(a1, (__int64)&v43);
    if ( (_BYTE)v6 )
    {
      v55 = "expected allockind value";
      v57.m128i_i16[4] = 259;
      sub_11FD800(v2, v41, (__int64)&v55, 1);
    }
    else
    {
      v52 = 0u;
      v53 = v43;
      v54.m128i_i64[0] = (__int64)",";
      v54.m128i_i64[1] = 1;
      v9 = sub_C931B0(v53.m128i_i64, ",", 1u, 0);
      if ( v9 == -1 )
      {
        v9 = v53.m128i_u64[1];
        v11 = v53.m128i_i64[0];
        v12 = 0;
        v13 = 0;
      }
      else
      {
        v10 = v9 + 1;
        v11 = v53.m128i_i64[0];
        if ( v9 + 1 > v53.m128i_i64[1] )
        {
          v10 = v53.m128i_i64[1];
          v12 = 0;
        }
        else
        {
          v12 = v53.m128i_i64[1] - v10;
        }
        v13 = v53.m128i_i64[0] + v10;
        if ( v9 > v53.m128i_i64[1] )
          v9 = v53.m128i_u64[1];
      }
      v52.m128i_i64[0] = v11;
      v53.m128i_i64[0] = v13;
      v53.m128i_i64[1] = v12;
      v52.m128i_i64[1] = v9;
      v48 = 0u;
      v49 = 0u;
      v50.m128i_i64[0] = (__int64)",";
      v50.m128i_i64[1] = 1;
      v14 = sub_C931B0(v49.m128i_i64, ",", 1u, 0);
      if ( v14 == -1 )
      {
        v14 = v49.m128i_u64[1];
        v16 = v49.m128i_i64[0];
        v17 = 0;
        v18 = 0;
      }
      else
      {
        v15 = v14 + 1;
        v16 = v49.m128i_i64[0];
        if ( v14 + 1 > v49.m128i_i64[1] )
        {
          v15 = v49.m128i_i64[1];
          v17 = 0;
        }
        else
        {
          v17 = v49.m128i_i64[1] - v15;
        }
        v18 = v49.m128i_i64[0] + v15;
        if ( v14 > v49.m128i_i64[1] )
          v14 = v49.m128i_u64[1];
      }
      v19 = _mm_loadu_si128(&v52);
      v48.m128i_i64[0] = v16;
      v20 = _mm_loadu_si128(&v53);
      v21 = _mm_loadu_si128(&v54);
      v48.m128i_i64[1] = v14;
      v49.m128i_i64[0] = v18;
      v49.m128i_i64[1] = v17;
      LOBYTE(v55) = v51;
      v56 = v19;
      v57 = v20;
      v58 = v21;
      if ( (char *)v54.m128i_i64[0] == &v51 )
      {
        v58.m128i_i64[1] = 1;
        v22 = &v55;
        v58.m128i_i64[0] = (__int64)&v55;
      }
      else
      {
        v22 = (const char **)v58.m128i_i64[0];
      }
      v23 = v47;
      v24 = _mm_loadu_si128(&v48);
      v25 = _mm_loadu_si128(&v49);
      v26 = _mm_loadu_si128(&v50);
      v59 = v47;
      v60 = v24;
      v61 = v25;
      v62 = v26;
      if ( (char *)v50.m128i_i64[0] == &v47 )
      {
        v62.m128i_i64[1] = 1;
        v62.m128i_i64[0] = (__int64)&v59;
        v27 = &v59;
      }
      else
      {
        v27 = (char *)v62.m128i_i64[0];
      }
      v28 = _mm_loadu_si128(&v56);
      v29 = _mm_loadu_si128(&v57);
      v47 = v51;
      v30 = _mm_loadu_si128(&v58);
      v48 = v28;
      v49 = v29;
      v50 = v30;
      if ( v22 == &v55 )
      {
        v50.m128i_i64[0] = (__int64)&v47;
        v50.m128i_i64[1] = 1;
      }
      v31 = _mm_loadu_si128(&v60);
      v32 = _mm_loadu_si128(&v61);
      v51 = v23;
      v33 = _mm_loadu_si128(&v62);
      v52 = v31;
      v53 = v32;
      v54 = v33;
      if ( v27 == &v59 )
      {
        v54.m128i_i64[0] = (__int64)&v51;
        v54.m128i_i64[1] = 1;
      }
      v34 = v48.m128i_u64[1];
      for ( i = v48.m128i_i64[0]; v52.m128i_i64[0] != i; v49.m128i_i64[1] = v38 )
      {
        switch ( v34 )
        {
          case 5uLL:
            if ( *(_DWORD *)i != 1869376609 || *(_BYTE *)(i + 4) != 99 )
              goto LABEL_29;
            *a2 |= 1uLL;
            break;
          case 7uLL:
            if ( *(_DWORD *)i == 1818322290 && *(_WORD *)(i + 4) == 28524 && *(_BYTE *)(i + 6) == 99 )
            {
              *a2 |= 2uLL;
            }
            else
            {
              if ( *(_DWORD *)i != 1734962273 || *(_WORD *)(i + 4) != 25966 || *(_BYTE *)(i + 6) != 100 )
              {
LABEL_29:
                v45[2] = i;
                v45[3] = v34;
                v45[0] = "unknown allockind ";
                v6 = 1;
                v46 = 1283;
                sub_11FD800(v2, v41, (__int64)v45, 1);
                goto LABEL_6;
              }
              *a2 |= 0x20uLL;
            }
            break;
          case 4uLL:
            if ( *(_DWORD *)i != 1701147238 )
              goto LABEL_29;
            *a2 |= 4uLL;
            break;
          case 0xDuLL:
            if ( *(_QWORD *)i != 0x616974696E696E75LL || *(_DWORD *)(i + 8) != 1702521196 || *(_BYTE *)(i + 12) != 100 )
              goto LABEL_29;
            *a2 |= 8uLL;
            break;
          default:
            if ( v34 != 6 || *(_DWORD *)i != 1869768058 || *(_WORD *)(i + 4) != 25701 )
              goto LABEL_29;
            *a2 |= 0x10uLL;
            break;
        }
        v36 = v50.m128i_i64[1];
        v34 = sub_C931B0(v49.m128i_i64, v50.m128i_i64[0], v50.m128i_u64[1], 0);
        if ( v34 == -1 )
        {
          v34 = v49.m128i_u64[1];
          i = v49.m128i_i64[0];
          v38 = 0;
          v39 = 0;
        }
        else
        {
          v37 = v34 + v36;
          i = v49.m128i_i64[0];
          if ( v37 > v49.m128i_i64[1] )
          {
            v37 = v49.m128i_u64[1];
            v38 = 0;
          }
          else
          {
            v38 = v49.m128i_i64[1] - v37;
          }
          v39 = v49.m128i_i64[0] + v37;
          if ( v34 > v49.m128i_i64[1] )
            v34 = v49.m128i_u64[1];
        }
        v48.m128i_i64[0] = i;
        v48.m128i_i64[1] = v34;
        v49.m128i_i64[0] = v39;
      }
      v40 = *(_QWORD *)(a1 + 232);
      if ( *(_DWORD *)(a1 + 240) == 13 )
      {
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        if ( !*a2 )
        {
          v6 = 1;
          v55 = "expected allockind value";
          v57.m128i_i16[4] = 259;
          sub_11FD800(v2, v41, (__int64)&v55, 1);
        }
      }
      else
      {
        v57.m128i_i16[4] = 259;
        v6 = 1;
        v55 = "expected ')'";
        sub_11FD800(v2, v40, (__int64)&v55, 1);
      }
    }
LABEL_6:
    if ( (_QWORD *)v43.m128i_i64[0] != v44 )
      j_j___libc_free_0(v43.m128i_i64[0], v44[0] + 1LL);
  }
  else
  {
    v57.m128i_i16[4] = 259;
    v6 = 1;
    v55 = "expected '('";
    sub_11FD800(v2, v5, (__int64)&v55, 1);
  }
  return v6;
}
