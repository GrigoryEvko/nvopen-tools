// Function: sub_2D507F0
// Address: 0x2d507f0
//
__int64 *__fastcall sub_2D507F0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v9; // r15
  unsigned int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // r13d
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  char *(*v17)(); // rax
  char v18; // al
  _QWORD *v19; // rsi
  char v20; // dl
  __m128i *v21; // rdi
  char v22; // al
  _QWORD *v23; // rsi
  char v24; // dl
  __m128i *v25; // rdi
  __m128i *v26; // rsi
  char v27; // dl
  __m128i v28; // xmm7
  __int64 v29; // r9
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rbx
  __m128i v34; // xmm1
  __m128i *v35; // rdi
  __int128 *v36; // rsi
  __m128i v37; // xmm3
  __m128i v38; // xmm6
  __m128i v39; // xmm7
  __m128i v40; // xmm5
  __m128i v41; // xmm5
  __int64 v42; // [rsp+0h] [rbp-1F0h]
  __int64 v43; // [rsp+8h] [rbp-1E8h]
  __int64 v44; // [rsp+10h] [rbp-1E0h]
  __int64 v45; // [rsp+18h] [rbp-1D8h]
  __int64 v46; // [rsp+20h] [rbp-1D0h]
  __int64 v47; // [rsp+28h] [rbp-1C8h]
  __int64 v48; // [rsp+38h] [rbp-1B8h] BYREF
  __m128i v49; // [rsp+40h] [rbp-1B0h] BYREF
  unsigned __int128 v50; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v51; // [rsp+60h] [rbp-190h]
  _QWORD v52[4]; // [rsp+70h] [rbp-180h] BYREF
  char v53; // [rsp+90h] [rbp-160h]
  char v54; // [rsp+91h] [rbp-15Fh]
  __m128i v55; // [rsp+A0h] [rbp-150h] BYREF
  __m128i v56; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v57; // [rsp+C0h] [rbp-130h]
  _QWORD v58[4]; // [rsp+D0h] [rbp-120h] BYREF
  __int16 v59; // [rsp+F0h] [rbp-100h]
  __m128i v60; // [rsp+100h] [rbp-F0h] BYREF
  __m128i v61; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v62; // [rsp+120h] [rbp-D0h]
  __m128i v63; // [rsp+130h] [rbp-C0h] BYREF
  __m128i v64; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+150h] [rbp-A0h]
  __m128i v66; // [rsp+160h] [rbp-90h] BYREF
  __m128i v67; // [rsp+170h] [rbp-80h] BYREF
  __int64 v68; // [rsp+180h] [rbp-70h]
  __m128i v69; // [rsp+190h] [rbp-60h] BYREF
  __m128i v70; // [rsp+1A0h] [rbp-50h]
  __int64 v71; // [rsp+1B0h] [rbp-40h]

  v11 = sub_C63BB0();
  v12 = *a2;
  v13 = v11;
  LOWORD(v65) = 259;
  v15 = v14;
  v63.m128i_i64[0] = (__int64)": ";
  v16 = *((unsigned int *)a2 + 13);
  v59 = 268;
  v48 = v16;
  v58[0] = &v48;
  v54 = 1;
  v53 = 3;
  v52[0] = " at line ";
  v17 = *(char *(**)())(*(_QWORD *)v12 + 16LL);
  if ( v17 == sub_C1E8B0 )
  {
    *((_QWORD *)&v50 + 1) = 14;
    LOWORD(v51) = 1283;
    v49.m128i_i64[0] = (__int64)"invalid profile ";
    *(_QWORD *)&v50 = "Unknown buffer";
    v18 = 3;
    goto LABEL_3;
  }
  v29 = (__int64)v17();
  v18 = v53;
  LOWORD(v51) = 1283;
  v49.m128i_i64[0] = (__int64)"invalid profile ";
  v50 = __PAIR128__(v30, v29);
  if ( !v53 )
  {
    LOWORD(v57) = 256;
    goto LABEL_22;
  }
  if ( v53 != 1 )
  {
LABEL_3:
    if ( v54 == 1 )
    {
      v9 = v52[1];
      v19 = (_QWORD *)v52[0];
    }
    else
    {
      v19 = v52;
      v18 = 2;
    }
    v56.m128i_i64[0] = (__int64)v19;
    v55.m128i_i64[0] = (__int64)&v49;
    v20 = v59;
    v56.m128i_i64[1] = v9;
    LOBYTE(v57) = 2;
    BYTE1(v57) = v18;
    if ( (_BYTE)v59 )
    {
      if ( (_BYTE)v59 != 1 )
        goto LABEL_7;
      goto LABEL_39;
    }
LABEL_22:
    LOWORD(v62) = 256;
    goto LABEL_23;
  }
  v34 = _mm_loadu_si128((const __m128i *)&v50);
  v20 = v59;
  v55 = _mm_loadu_si128(&v49);
  v57 = v51;
  v56 = v34;
  if ( !(_BYTE)v59 )
    goto LABEL_22;
  if ( (_BYTE)v59 != 1 )
  {
    if ( BYTE1(v57) == 1 )
    {
      v47 = v55.m128i_i64[1];
      v21 = (__m128i *)v55.m128i_i64[0];
      v22 = 3;
LABEL_8:
      if ( HIBYTE(v59) == 1 )
      {
        v23 = (_QWORD *)v58[0];
        v46 = v58[1];
      }
      else
      {
        v23 = v58;
        v20 = 2;
      }
      BYTE1(v62) = v20;
      v24 = v65;
      v60.m128i_i64[0] = (__int64)v21;
      v60.m128i_i64[1] = v47;
      v61.m128i_i64[0] = (__int64)v23;
      v61.m128i_i64[1] = v46;
      LOBYTE(v62) = v22;
      if ( (_BYTE)v65 )
      {
LABEL_11:
        if ( v24 == 1 )
        {
          v40 = _mm_loadu_si128(&v61);
          v22 = v62;
          v66 = _mm_loadu_si128(&v60);
          v68 = v62;
          v67 = v40;
          if ( !(_BYTE)v62 )
            goto LABEL_24;
        }
        else
        {
          if ( BYTE1(v62) == 1 )
          {
            v45 = v60.m128i_i64[1];
            v25 = (__m128i *)v60.m128i_i64[0];
          }
          else
          {
            v25 = &v60;
            v22 = 2;
          }
          if ( BYTE1(v65) == 1 )
          {
            v44 = v63.m128i_i64[1];
            v26 = (__m128i *)v63.m128i_i64[0];
          }
          else
          {
            v26 = &v63;
            v24 = 2;
          }
          v66.m128i_i64[0] = (__int64)v25;
          v67.m128i_i64[0] = (__int64)v26;
          v66.m128i_i64[1] = v45;
          LOBYTE(v68) = v22;
          v67.m128i_i64[1] = v44;
          BYTE1(v68) = v24;
        }
        goto LABEL_17;
      }
LABEL_23:
      LOWORD(v68) = 256;
      goto LABEL_24;
    }
LABEL_7:
    v21 = &v55;
    v22 = 2;
    goto LABEL_8;
  }
LABEL_39:
  v37 = _mm_loadu_si128(&v56);
  v22 = v57;
  v60 = _mm_loadu_si128(&v55);
  v62 = v57;
  v61 = v37;
  if ( !(_BYTE)v57 )
    goto LABEL_23;
  v24 = v65;
  if ( !(_BYTE)v65 )
    goto LABEL_23;
  if ( (_BYTE)v57 != 1 )
    goto LABEL_11;
  v38 = _mm_loadu_si128(&v63);
  v39 = _mm_loadu_si128(&v64);
  v68 = v65;
  v22 = v65;
  v66 = v38;
  v67 = v39;
LABEL_17:
  v27 = a9;
  if ( (_BYTE)a9 )
  {
    if ( v22 == 1 )
    {
      v28 = _mm_loadu_si128((const __m128i *)&a8);
      v69 = _mm_loadu_si128((const __m128i *)&a7);
      v71 = a9;
      v70 = v28;
    }
    else if ( (_BYTE)a9 == 1 )
    {
      v41 = _mm_loadu_si128(&v67);
      v69 = _mm_loadu_si128(&v66);
      v71 = v68;
      v70 = v41;
    }
    else
    {
      if ( BYTE1(v68) == 1 )
      {
        v43 = v66.m128i_i64[1];
        v35 = (__m128i *)v66.m128i_i64[0];
      }
      else
      {
        v35 = &v66;
        v22 = 2;
      }
      if ( BYTE1(a9) == 1 )
      {
        v42 = *((_QWORD *)&a7 + 1);
        v36 = (__int128 *)a7;
      }
      else
      {
        v36 = &a7;
        v27 = 2;
      }
      v69.m128i_i64[0] = (__int64)v35;
      v70.m128i_i64[0] = (__int64)v36;
      v69.m128i_i64[1] = v43;
      LOBYTE(v71) = v22;
      v70.m128i_i64[1] = v42;
      BYTE1(v71) = v27;
    }
    goto LABEL_25;
  }
LABEL_24:
  LOWORD(v71) = 256;
LABEL_25:
  v31 = sub_22077B0(0x40u);
  v32 = v31;
  if ( v31 )
    sub_C63EB0(v31, (__int64)&v69, v13, v15);
  *a1 = v32 | 1;
  return a1;
}
