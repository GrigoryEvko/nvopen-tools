// Function: sub_2A43B20
// Address: 0x2a43b20
//
__int64 __fastcall sub_2A43B20(__int64 a1)
{
  __m128i *v1; // r15
  __m128i *v2; // rdi
  __int64 (__fastcall *v3)(__int64); // rax
  __m128i *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rdi
  char v7; // al
  __m128i *v8; // rdx
  char v9; // al
  __m128i *v10; // rsi
  char v11; // cl
  __m128i *v12; // rdx
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __m128i *v15; // r14
  __m128i *v16; // rdi
  __int64 (__fastcall *v17)(__int64); // rax
  __m128i *v18; // rbx
  __int64 v19; // rdi
  __int32 v20; // eax
  char v21; // cl
  __int64 v22; // r15
  __m128i *v23; // rcx
  char v24; // al
  char v25; // al
  __m128i *v26; // rdi
  char v27; // si
  __m128i *v28; // rcx
  __int32 v29; // r13d
  __m128i v30; // xmm7
  __m128i v31; // xmm0
  __m128i v32; // xmm1
  __m128i v34; // xmm7
  unsigned __int64 v36; // [rsp+8h] [rbp-228h]
  unsigned __int64 v37; // [rsp+18h] [rbp-218h]
  unsigned __int64 v38; // [rsp+28h] [rbp-208h]
  __int64 v39; // [rsp+30h] [rbp-200h]
  __int64 v40; // [rsp+38h] [rbp-1F8h]
  __int64 v41; // [rsp+40h] [rbp-1F0h]
  __int64 v42; // [rsp+48h] [rbp-1E8h]
  __int64 v43; // [rsp+50h] [rbp-1E0h]
  __int64 v44; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 v45; // [rsp+60h] [rbp-1D0h]
  unsigned __int8 *v46; // [rsp+60h] [rbp-1D0h]
  __int32 v47; // [rsp+6Ch] [rbp-1C4h]
  unsigned __int64 v48; // [rsp+78h] [rbp-1B8h]
  __m128i v49; // [rsp+80h] [rbp-1B0h] BYREF
  __m128i v50; // [rsp+90h] [rbp-1A0h]
  __int64 v51; // [rsp+A0h] [rbp-190h] BYREF
  _QWORD *v52; // [rsp+A8h] [rbp-188h]
  __int64 v53; // [rsp+B0h] [rbp-180h]
  _BYTE v54[24]; // [rsp+B8h] [rbp-178h] BYREF
  __m128i v55; // [rsp+D0h] [rbp-160h] BYREF
  __m128i v56; // [rsp+E0h] [rbp-150h]
  __int64 v57; // [rsp+F0h] [rbp-140h]
  __m128i v58; // [rsp+100h] [rbp-130h] BYREF
  __m128i v59; // [rsp+110h] [rbp-120h]
  unsigned __int64 v60; // [rsp+120h] [rbp-110h]
  __m128i v61; // [rsp+130h] [rbp-100h] BYREF
  __m128i v62; // [rsp+140h] [rbp-F0h] BYREF
  __int64 v63; // [rsp+150h] [rbp-E0h]
  __m128i v64; // [rsp+160h] [rbp-D0h] BYREF
  __m128i v65; // [rsp+170h] [rbp-C0h] BYREF
  unsigned __int64 v66; // [rsp+180h] [rbp-B0h] BYREF
  __m128i v67; // [rsp+190h] [rbp-A0h] BYREF
  __m128i v68; // [rsp+1A0h] [rbp-90h] BYREF
  __int64 v69; // [rsp+1B0h] [rbp-80h] BYREF
  __m128i v70; // [rsp+1C0h] [rbp-70h] BYREF
  __m128i v71; // [rsp+1D0h] [rbp-60h] BYREF
  unsigned __int64 v72; // [rsp+1E0h] [rbp-50h]
  unsigned __int64 v73; // [rsp+1E8h] [rbp-48h]
  unsigned __int64 v74; // [rsp+1F0h] [rbp-40h]
  unsigned __int64 v75; // [rsp+1F8h] [rbp-38h]

  v52 = v54;
  v51 = a1;
  v53 = 0;
  v54[0] = 0;
  sub_BA9600(&v70, a1);
  v45 = 0;
  v47 = 0;
  v38 = v72;
  v49 = _mm_loadu_si128(&v70);
  v48 = v73;
  v50 = _mm_loadu_si128(&v71);
  v36 = v74;
  v37 = v75;
  while ( __PAIR128__(v48, v38) != *(_OWORD *)&v49 || *(_OWORD *)&v50 != __PAIR128__(v37, v36) )
  {
    v1 = &v67;
    v2 = &v49;
    v68.m128i_i64[1] = 0;
    v68.m128i_i64[0] = (__int64)sub_25AC5E0;
    v3 = sub_25AC5C0;
    v4 = &v67;
    if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
      goto LABEL_5;
    while ( 1 )
    {
      v3 = *(__int64 (__fastcall **)(__int64))((char *)v3 + v2->m128i_i64[0] - 1);
LABEL_5:
      v5 = v3((__int64)v2);
      if ( v5 )
        break;
      while ( 1 )
      {
        if ( ++v1 == (__m128i *)&v69 )
LABEL_67:
          BUG();
        v6 = v4[1].m128i_i64[1];
        v3 = (__int64 (__fastcall *)(__int64))v4[1].m128i_i64[0];
        v4 = v1;
        v2 = (__m128i *)((char *)&v49 + v6);
        if ( ((unsigned __int8)v3 & 1) != 0 )
          break;
        v5 = v3((__int64)v2);
        if ( v5 )
          goto LABEL_9;
      }
    }
LABEL_9:
    if ( (*(_BYTE *)(v5 + 7) & 0x10) != 0 )
      goto LABEL_27;
    v46 = (unsigned __int8 *)v5;
    LOWORD(v60) = 266;
    v58.m128i_i32[0] = v47;
    v64.m128i_i64[0] = (__int64)".";
    LOWORD(v66) = 259;
    v68.m128i_i64[0] = (__int64)sub_2A43790(&v51);
    v7 = v66;
    v67.m128i_i64[0] = (__int64)"anon.";
    LOWORD(v69) = 1027;
    if ( (_BYTE)v66 )
    {
      if ( (_BYTE)v66 == 1 )
      {
        v13 = _mm_loadu_si128(&v67);
        v14 = _mm_loadu_si128(&v68);
        v63 = v69;
        v9 = v60;
        v61 = v13;
        v62 = v14;
        if ( (_BYTE)v60 )
        {
          if ( (_BYTE)v60 != 1 )
          {
            if ( BYTE1(v63) == 1 )
            {
              v42 = v61.m128i_i64[1];
              v10 = (__m128i *)v61.m128i_i64[0];
              v11 = 3;
            }
            else
            {
LABEL_16:
              v10 = &v61;
              v11 = 2;
            }
            if ( BYTE1(v60) == 1 )
            {
              v41 = v58.m128i_i64[1];
              v12 = (__m128i *)v58.m128i_i64[0];
            }
            else
            {
              v12 = &v58;
              v9 = 2;
            }
            v55.m128i_i64[0] = (__int64)v10;
            v56.m128i_i64[0] = (__int64)v12;
            v55.m128i_i64[1] = v42;
            LOBYTE(v57) = v11;
            v56.m128i_i64[1] = v41;
            BYTE1(v57) = v9;
            goto LABEL_26;
          }
          goto LABEL_52;
        }
      }
      else
      {
        if ( BYTE1(v66) == 1 )
        {
          v43 = v64.m128i_i64[1];
          v8 = (__m128i *)v64.m128i_i64[0];
        }
        else
        {
          v8 = &v64;
          v7 = 2;
        }
        BYTE1(v63) = v7;
        v9 = v60;
        v61.m128i_i64[0] = (__int64)&v67;
        v62.m128i_i64[0] = (__int64)v8;
        v62.m128i_i64[1] = v43;
        LOBYTE(v63) = 2;
        if ( (_BYTE)v60 )
        {
          if ( (_BYTE)v60 != 1 )
            goto LABEL_16;
LABEL_52:
          v30 = _mm_loadu_si128(&v62);
          v55 = _mm_loadu_si128(&v61);
          v57 = v63;
          v56 = v30;
          goto LABEL_26;
        }
      }
    }
    else
    {
      LOWORD(v63) = 256;
    }
    LOWORD(v57) = 256;
LABEL_26:
    sub_BD6B50(v46, (const char **)&v55);
    ++v47;
    v45 = 1;
LABEL_27:
    v15 = &v64;
    v65.m128i_i64[1] = 0;
    v16 = &v49;
    v65.m128i_i64[0] = (__int64)sub_25AC590;
    v17 = sub_25AC560;
    v18 = &v64;
    if ( ((unsigned __int8)sub_25AC560 & 1) == 0 )
      goto LABEL_29;
LABEL_28:
    v17 = *(__int64 (__fastcall **)(__int64))((char *)v17 + v16->m128i_i64[0] - 1);
LABEL_29:
    while ( !(unsigned __int8)v17((__int64)v16) )
    {
      if ( ++v15 == (__m128i *)&v66 )
        goto LABEL_67;
      v19 = v18[1].m128i_i64[1];
      v17 = (__int64 (__fastcall *)(__int64))v18[1].m128i_i64[0];
      v18 = v15;
      v16 = (__m128i *)((char *)&v49 + v19);
      if ( ((unsigned __int8)v17 & 1) != 0 )
        goto LABEL_28;
    }
  }
  if ( a1 + 40 == *(_QWORD *)(a1 + 48) )
    goto LABEL_58;
  v20 = v47;
  v21 = v45;
  v22 = *(_QWORD *)(a1 + 48);
  while ( 2 )
  {
    if ( !v22 )
      BUG();
    if ( (*(_BYTE *)(v22 - 41) & 0x10) == 0 )
    {
      v29 = v20 + 1;
      v61.m128i_i32[0] = v20;
      LOWORD(v63) = 266;
      v67.m128i_i64[0] = (__int64)".";
      LOWORD(v69) = 259;
      v71.m128i_i64[0] = (__int64)sub_2A43790(&v51);
      v24 = v69;
      v70.m128i_i64[0] = (__int64)"anon.";
      LOWORD(v72) = 1027;
      if ( !(_BYTE)v69 )
      {
        LOWORD(v66) = 256;
        goto LABEL_51;
      }
      if ( (_BYTE)v69 != 1 )
      {
        if ( BYTE1(v69) == 1 )
        {
          v44 = v67.m128i_i64[1];
          v23 = (__m128i *)v67.m128i_i64[0];
        }
        else
        {
          v23 = &v67;
          v24 = 2;
        }
        v65.m128i_i64[0] = (__int64)v23;
        BYTE1(v66) = v24;
        v25 = v63;
        v64.m128i_i64[0] = (__int64)&v70;
        v65.m128i_i64[1] = v44;
        LOBYTE(v66) = 2;
        if ( (_BYTE)v63 )
        {
          if ( (_BYTE)v63 != 1 )
            goto LABEL_41;
LABEL_61:
          v34 = _mm_loadu_si128(&v65);
          v58 = _mm_loadu_si128(&v64);
          v60 = v66;
          v59 = v34;
          goto LABEL_45;
        }
LABEL_51:
        LOWORD(v60) = 256;
        goto LABEL_45;
      }
      v31 = _mm_loadu_si128(&v70);
      v32 = _mm_loadu_si128(&v71);
      v66 = v72;
      v25 = v63;
      v64 = v31;
      v65 = v32;
      if ( !(_BYTE)v63 )
        goto LABEL_51;
      if ( (_BYTE)v63 == 1 )
        goto LABEL_61;
      if ( BYTE1(v66) == 1 )
      {
        v26 = (__m128i *)v64.m128i_i64[0];
        v27 = 3;
        v40 = v64.m128i_i64[1];
      }
      else
      {
LABEL_41:
        v26 = &v64;
        v27 = 2;
      }
      if ( BYTE1(v63) == 1 )
      {
        v39 = v61.m128i_i64[1];
        v28 = (__m128i *)v61.m128i_i64[0];
      }
      else
      {
        v28 = &v61;
        v25 = 2;
      }
      v59.m128i_i64[0] = (__int64)v28;
      v58.m128i_i64[0] = (__int64)v26;
      v58.m128i_i64[1] = v40;
      v59.m128i_i64[1] = v39;
      LOBYTE(v60) = v27;
      BYTE1(v60) = v25;
LABEL_45:
      sub_BD6B50((unsigned __int8 *)(v22 - 48), (const char **)&v58);
      v20 = v29;
      v21 = 1;
    }
    v22 = *(_QWORD *)(v22 + 8);
    if ( a1 + 40 != v22 )
      continue;
    break;
  }
  v45 = v21;
LABEL_58:
  if ( v52 != (_QWORD *)v54 )
    j_j___libc_free_0((unsigned __int64)v52);
  return v45;
}
