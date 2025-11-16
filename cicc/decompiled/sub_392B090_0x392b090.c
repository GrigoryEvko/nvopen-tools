// Function: sub_392B090
// Address: 0x392b090
//
__int64 __fastcall sub_392B090(__int64 a1, __int64 a2)
{
  char v4; // r9
  _BYTE *v5; // rsi
  char v6; // al
  _BYTE *v7; // rcx
  _BYTE *v8; // r12
  char v9; // al
  unsigned int v10; // r15d
  __int64 v11; // rax
  _BYTE *v12; // rbx
  __int64 v13; // rax
  const char *v14; // r12
  size_t v15; // rax
  unsigned int v16; // r15d
  _QWORD *v17; // rax
  unsigned __int64 v18; // rsi
  char *v19; // rax
  unsigned int v20; // edx
  const char *v21; // r12
  unsigned int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rax
  char v26; // al
  _BYTE *v27; // rdx
  unsigned __int8 v28; // al
  _BYTE *v29; // rsi
  _BYTE *v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // rax
  __m128i v35; // xmm0
  unsigned __int64 v36; // rdi
  unsigned int v37; // ebx
  _BYTE *v38; // rax
  __int64 v39; // rdx
  char *v40; // rax
  char v41; // dl
  unsigned __int64 v42; // rdi
  char *v44; // rsi
  char *i; // rax
  char v46; // dl
  __int64 v47; // rax
  char *v48; // rdx
  char v49; // al
  const char *v50; // r15
  size_t v51; // rax
  unsigned int v52; // r12d
  _QWORD *v53; // rax
  unsigned __int64 v54; // rsi
  char *v55; // rax
  unsigned int v56; // edx
  const char *v57; // r15
  unsigned int v58; // eax
  unsigned int v59; // edx
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // rax
  unsigned int v63; // ebx
  __int64 v64; // rax
  char *v65; // rax
  char v66; // dl
  const char *v67; // r12
  size_t v68; // rax
  unsigned int v69; // r15d
  _QWORD *v70; // rax
  unsigned __int64 v71; // rsi
  char *v72; // rax
  unsigned int v73; // edx
  const char *v74; // r12
  unsigned int v75; // eax
  unsigned int v76; // edx
  __int64 v77; // rcx
  _OWORD *v78; // rax
  __m128i v79; // xmm0
  __int64 v80; // rdx
  __int64 v81; // rax
  __m128i v82; // xmm0
  char *v83; // rax
  char v84; // dl
  __int64 v85; // rax
  _OWORD *v86; // rax
  __m128i si128; // xmm0
  __int64 v88; // [rsp+0h] [rbp-90h]
  __int64 v89; // [rsp+8h] [rbp-88h]
  __int64 v90; // [rsp+18h] [rbp-78h] BYREF
  __m128i v91; // [rsp+20h] [rbp-70h] BYREF
  __m128i v92; // [rsp+30h] [rbp-60h] BYREF
  __m128i v93; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v94[8]; // [rsp+50h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a2 + 170);
  v5 = *(_BYTE **)(a2 + 144);
  v6 = *(v5 - 1);
  if ( !v4 || (unsigned int)(v6 - 48) > 9 )
  {
LABEL_20:
    if ( v6 == 48 && *v5 != 46 )
    {
      v26 = *v5 & 0xDF;
      if ( v4 || v26 != 66 )
      {
        if ( v26 != 88 )
        {
          v91.m128i_i32[2] = 128;
          sub_16A4EF0((__int64)&v91, 0, 1);
          v63 = sub_392A370((__int64 *)(a2 + 144), 8u);
          v64 = *(_QWORD *)(a2 + 144);
          v92.m128i_i64[0] = *(_QWORD *)(a2 + 104);
          v92.m128i_i64[1] = v64 - v92.m128i_i64[0];
          if ( (unsigned __int8)sub_16D2BE0(&v92, v63, (unsigned __int64 *)&v91) )
          {
            v67 = "invalid hexdecimal number";
            if ( v63 != 16 )
              v67 = "invalid octal number";
            v93.m128i_i64[0] = (__int64)v94;
            v68 = strlen(v67);
            v69 = v68;
            v90 = v68;
            v70 = (_QWORD *)sub_22409D0((__int64)&v93, (unsigned __int64 *)&v90, 0);
            v93.m128i_i64[0] = (__int64)v70;
            v94[0] = v90;
            v71 = (unsigned __int64)(v70 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *v70 = *(_QWORD *)v67;
            *(_QWORD *)((char *)v70 + v69 - 8) = *(_QWORD *)&v67[v69 - 8];
            v72 = (char *)v70 - v71;
            v73 = v69 + (_DWORD)v72;
            v74 = (const char *)(v67 - v72);
            v75 = 0;
            v76 = v73 & 0xFFFFFFF8;
            do
            {
              v77 = v75;
              v75 += 8;
              *(_QWORD *)(v71 + v77) = *(_QWORD *)&v74[v77];
            }
            while ( v75 < v76 );
            v93.m128i_i64[1] = v90;
            *(_BYTE *)(v93.m128i_i64[0] + v90) = 0;
            sub_392A760(a1, (_QWORD *)a2, *(_QWORD *)(a2 + 104), (unsigned __int64 *)&v93);
            if ( (_QWORD *)v93.m128i_i64[0] != v94 )
              j_j___libc_free_0(v93.m128i_u64[0]);
          }
          else
          {
            v65 = *(char **)(a2 + 144);
            if ( v63 == 16 )
              *(_QWORD *)(a2 + 144) = ++v65;
            v66 = *v65;
            if ( *v65 == 85 )
            {
              *(_QWORD *)(a2 + 144) = v65 + 1;
              v66 = *++v65;
            }
            if ( v66 == 76 )
            {
              *(_QWORD *)(a2 + 144) = v65 + 1;
              if ( v65[1] == 76 )
                *(_QWORD *)(a2 + 144) = v65 + 2;
            }
            sub_392A2A0(a1, v92.m128i_i64[0], v92.m128i_i64[1], (__int64)&v91);
          }
          if ( v91.m128i_i32[2] > 0x40u )
          {
            v42 = v91.m128i_i64[0];
            if ( v91.m128i_i64[0] )
              goto LABEL_45;
          }
          return a1;
        }
        v44 = v5 + 1;
        *(_QWORD *)(a2 + 144) = v44;
        for ( i = v44; ; *(_QWORD *)(a2 + 144) = i )
        {
          v46 = *i;
          if ( (unsigned __int8)(*i - 48) > 9u && (unsigned __int8)((v46 & 0xDF) - 65) > 5u )
            break;
          ++i;
        }
        if ( (v46 & 0xDF) == 0x50 || v46 == 46 )
        {
          sub_392A8D0(a1, (_QWORD *)a2, v44 == i);
          return a1;
        }
        if ( v44 != i )
        {
          v92.m128i_i32[2] = 128;
          sub_16A4EF0((__int64)&v92, 0, 0);
          v47 = *(_QWORD *)(a2 + 144);
          v93.m128i_i64[0] = *(_QWORD *)(a2 + 104);
          v93.m128i_i64[1] = v47 - v93.m128i_i64[0];
          if ( !(unsigned __int8)sub_16D2BE0(&v93, 0, (unsigned __int64 *)&v92) )
          {
            v48 = *(char **)(a2 + 144);
            v49 = *v48;
            if ( !*(_BYTE *)(a2 + 170) && (v49 & 0xDF) == 0x48 )
            {
              *(_QWORD *)(a2 + 144) = v48 + 1;
              v49 = *++v48;
            }
            if ( v49 == 85 )
            {
              *(_QWORD *)(a2 + 144) = v48 + 1;
              v49 = *++v48;
            }
            if ( v49 == 76 )
            {
              *(_QWORD *)(a2 + 144) = v48 + 1;
              if ( v48[1] == 76 )
              {
                v48 += 2;
                *(_QWORD *)(a2 + 144) = v48;
              }
              else
              {
                ++v48;
              }
            }
            sub_392A2A0(a1, *(_QWORD *)(a2 + 104), (__int64)&v48[-*(_QWORD *)(a2 + 104)], (__int64)&v92);
            goto LABEL_43;
          }
          v93.m128i_i64[0] = (__int64)v94;
          v91.m128i_i64[0] = 26;
          v86 = (_OWORD *)sub_22409D0((__int64)&v93, (unsigned __int64 *)&v91, 0);
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
          v93.m128i_i64[0] = (__int64)v86;
          v94[0] = v91.m128i_i64[0];
          qmemcpy(v86 + 1, "mal number", 10);
          *v86 = si128;
          goto LABEL_32;
        }
        v92.m128i_i64[0] = 26;
        v93.m128i_i64[0] = (__int64)v94;
        v78 = (_OWORD *)sub_22409D0((__int64)&v93, (unsigned __int64 *)&v92, 0);
        v79 = _mm_load_si128((const __m128i *)&xmmword_3F900A0);
        v93.m128i_i64[0] = (__int64)v78;
        v94[0] = v92.m128i_i64[0];
        qmemcpy(v78 + 1, "mal number", 10);
        *v78 = v79;
        v93.m128i_i64[1] = v92.m128i_i64[0];
        *(_BYTE *)(v93.m128i_i64[0] + v92.m128i_i64[0]) = 0;
        v80 = *(_QWORD *)(a2 + 144) - 2LL;
      }
      else
      {
        v27 = v5 + 1;
        *(_QWORD *)(a2 + 144) = v5 + 1;
        v28 = v5[1] - 48;
        if ( v28 > 9u )
        {
          v85 = *(_QWORD *)(a2 + 104);
          *(_QWORD *)(a2 + 144) = v5;
          *(_DWORD *)a1 = 4;
          *(_QWORD *)(a1 + 8) = v85;
          *(_QWORD *)(a1 + 16) = &v5[-v85];
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
          return a1;
        }
        v29 = v5 + 2;
        if ( v28 <= 1u )
        {
          do
          {
            v30 = v29;
            *(_QWORD *)(a2 + 144) = v29++;
          }
          while ( (unsigned __int8)(*v30 - 48) <= 1u );
          if ( v27 != v30 )
          {
            v31 = *(_QWORD *)(a2 + 104);
            v92.m128i_i32[2] = 128;
            v89 = v31;
            v32 = (unsigned __int64)&v30[-v31];
            v88 = (__int64)&v30[-v31];
            sub_16A4EF0((__int64)&v92, 0, 1);
            v33 = 0;
            if ( v32 > 1 )
            {
              v33 = v32 - 2;
              v30 = (_BYTE *)(v89 + 2);
            }
            v93.m128i_i64[0] = (__int64)v30;
            v93.m128i_i64[1] = v33;
            if ( !(unsigned __int8)sub_16D2BE0(&v93, 2u, (unsigned __int64 *)&v92) )
              goto LABEL_100;
            v93.m128i_i64[0] = (__int64)v94;
            v91.m128i_i64[0] = 21;
            v34 = sub_22409D0((__int64)&v93, (unsigned __int64 *)&v91, 0);
            v35 = _mm_load_si128((const __m128i *)&xmmword_3F90120);
            v93.m128i_i64[0] = v34;
            v94[0] = v91.m128i_i64[0];
            *(_DWORD *)(v34 + 16) = 1700949365;
            *(_BYTE *)(v34 + 20) = 114;
            *(__m128i *)v34 = v35;
LABEL_32:
            v93.m128i_i64[1] = v91.m128i_i64[0];
            *(_BYTE *)(v93.m128i_i64[0] + v91.m128i_i64[0]) = 0;
            sub_392A760(a1, (_QWORD *)a2, *(_QWORD *)(a2 + 104), (unsigned __int64 *)&v93);
            v36 = v93.m128i_i64[0];
            if ( (_QWORD *)v93.m128i_i64[0] == v94 )
              goto LABEL_43;
            goto LABEL_71;
          }
        }
        v92.m128i_i64[0] = 21;
        v93.m128i_i64[0] = (__int64)v94;
        v81 = sub_22409D0((__int64)&v93, (unsigned __int64 *)&v92, 0);
        v82 = _mm_load_si128((const __m128i *)&xmmword_3F90120);
        v93.m128i_i64[0] = v81;
        v94[0] = v92.m128i_i64[0];
        *(_DWORD *)(v81 + 16) = 1700949365;
        *(_BYTE *)(v81 + 20) = 114;
        *(__m128i *)v81 = v82;
        v93.m128i_i64[1] = v92.m128i_i64[0];
        *(_BYTE *)(v93.m128i_i64[0] + v92.m128i_i64[0]) = 0;
        v80 = *(_QWORD *)(a2 + 104);
      }
      sub_392A760(a1, (_QWORD *)a2, v80, (unsigned __int64 *)&v93);
      if ( (_QWORD *)v93.m128i_i64[0] != v94 )
        j_j___libc_free_0(v93.m128i_u64[0]);
      return a1;
    }
    v37 = sub_392A370((__int64 *)(a2 + 144), 0xAu);
    if ( v37 == 16 )
    {
      v61 = *(_QWORD *)(a2 + 104);
      v62 = *(_QWORD *)(a2 + 144);
      v92.m128i_i32[2] = 128;
      v50 = "invalid hexdecimal number";
      v91.m128i_i64[0] = v61;
      v91.m128i_i64[1] = v62 - v61;
      sub_16A4EF0((__int64)&v92, 0, 1);
      if ( !(unsigned __int8)sub_16D2BE0(&v91, 0x10u, (unsigned __int64 *)&v92) )
      {
        v40 = *(char **)(a2 + 144);
        goto LABEL_74;
      }
    }
    else
    {
      v38 = *(_BYTE **)(a2 + 144);
      if ( *v38 == 46 || *v38 == 101 )
      {
        *(_QWORD *)(a2 + 144) = v38 + 1;
        sub_392A800(a1, a2);
        return a1;
      }
      v39 = *(_QWORD *)(a2 + 104);
      v92.m128i_i32[2] = 128;
      v91.m128i_i64[0] = v39;
      v91.m128i_i64[1] = (__int64)&v38[-v39];
      sub_16A4EF0((__int64)&v92, 0, 1);
      if ( !(unsigned __int8)sub_16D2BE0(&v91, v37, (unsigned __int64 *)&v92) )
      {
        v40 = *(char **)(a2 + 144);
        if ( v37 != 2 )
        {
LABEL_39:
          v41 = *v40;
          if ( *v40 == 85 )
          {
            *(_QWORD *)(a2 + 144) = v40 + 1;
            v41 = *++v40;
          }
          if ( v41 == 76 )
          {
            *(_QWORD *)(a2 + 144) = v40 + 1;
            if ( v40[1] == 76 )
              *(_QWORD *)(a2 + 144) = v40 + 2;
          }
          sub_392A2A0(a1, v91.m128i_i64[0], v91.m128i_i64[1], (__int64)&v92);
          goto LABEL_43;
        }
LABEL_74:
        *(_QWORD *)(a2 + 144) = ++v40;
        goto LABEL_39;
      }
      v50 = "invalid decimal number";
    }
    v93.m128i_i64[0] = (__int64)v94;
    v51 = strlen(v50);
    v52 = v51;
    v90 = v51;
    v53 = (_QWORD *)sub_22409D0((__int64)&v93, (unsigned __int64 *)&v90, 0);
    v93.m128i_i64[0] = (__int64)v53;
    v94[0] = v90;
    v54 = (unsigned __int64)(v53 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v53 = *(_QWORD *)v50;
    *(_QWORD *)((char *)v53 + v52 - 8) = *(_QWORD *)&v50[v52 - 8];
    v55 = (char *)v53 - v54;
    v56 = v52 + (_DWORD)v55;
    v57 = (const char *)(v50 - v55);
    v58 = 0;
    v59 = v56 & 0xFFFFFFF8;
    do
    {
      v60 = v58;
      v58 += 8;
      *(_QWORD *)(v54 + v60) = *(_QWORD *)&v57[v60];
    }
    while ( v58 < v59 );
    v25 = v90;
    goto LABEL_70;
  }
  v7 = v5 - 1;
  v8 = v5;
  if ( (unsigned __int8)(v6 - 48) <= 1u )
    v7 = 0;
  while ( 1 )
  {
    if ( (unsigned __int8)(*v8 - 48) > 9u )
    {
      v9 = *v8 & 0xDF;
      if ( (unsigned __int8)(v9 - 65) > 5u )
        break;
    }
    if ( (unsigned __int8)(*v8 - 48) > 1u && !v7 )
      v7 = v8;
    *(_QWORD *)(a2 + 144) = ++v8;
  }
  if ( v9 == 72 )
  {
    ++v8;
    v10 = 16;
    *(_QWORD *)(a2 + 144) = v8;
    goto LABEL_9;
  }
  if ( !v7 || v8 != v7 + 1 || (*v7 & 0xDF) != 0x42 )
  {
    *(_QWORD *)(a2 + 144) = v5;
    v6 = *(v5 - 1);
    goto LABEL_20;
  }
  v10 = 2;
LABEL_9:
  v11 = *(_QWORD *)(a2 + 104);
  v92.m128i_i32[2] = 128;
  v89 = v11;
  v12 = &v8[-v11];
  v88 = (__int64)&v8[-v11];
  sub_16A4EF0((__int64)&v92, 0, 1);
  v13 = (__int64)(v12 - 1);
  if ( !v12 )
    v13 = 0;
  v93.m128i_i64[0] = v89;
  v93.m128i_i64[1] = v13;
  if ( !(unsigned __int8)sub_16D2BE0(&v93, v10, (unsigned __int64 *)&v92) )
  {
LABEL_100:
    v83 = *(char **)(a2 + 144);
    v84 = *v83;
    if ( *v83 == 85 )
    {
      *(_QWORD *)(a2 + 144) = v83 + 1;
      v84 = *++v83;
    }
    if ( v84 == 76 )
    {
      *(_QWORD *)(a2 + 144) = v83 + 1;
      if ( v83[1] == 76 )
        *(_QWORD *)(a2 + 144) = v83 + 2;
    }
    sub_392A2A0(a1, v89, v88, (__int64)&v92);
    goto LABEL_43;
  }
  v14 = "invalid binary number";
  if ( v10 != 2 )
    v14 = "invalid hexdecimal number";
  v93.m128i_i64[0] = (__int64)v94;
  v15 = strlen(v14);
  v16 = v15;
  v91.m128i_i64[0] = v15;
  v17 = (_QWORD *)sub_22409D0((__int64)&v93, (unsigned __int64 *)&v91, 0);
  v93.m128i_i64[0] = (__int64)v17;
  v94[0] = v91.m128i_i64[0];
  v18 = (unsigned __int64)(v17 + 1) & 0xFFFFFFFFFFFFFFF8LL;
  *v17 = *(_QWORD *)v14;
  *(_QWORD *)((char *)v17 + v16 - 8) = *(_QWORD *)&v14[v16 - 8];
  v19 = (char *)v17 - v18;
  v20 = v16 + (_DWORD)v19;
  v21 = (const char *)(v14 - v19);
  v22 = 0;
  v23 = v20 & 0xFFFFFFF8;
  do
  {
    v24 = v22;
    v22 += 8;
    *(_QWORD *)(v18 + v24) = *(_QWORD *)&v21[v24];
  }
  while ( v22 < v23 );
  v25 = v91.m128i_i64[0];
LABEL_70:
  v93.m128i_i64[1] = v25;
  *(_BYTE *)(v93.m128i_i64[0] + v25) = 0;
  sub_392A760(a1, (_QWORD *)a2, *(_QWORD *)(a2 + 104), (unsigned __int64 *)&v93);
  v36 = v93.m128i_i64[0];
  if ( (_QWORD *)v93.m128i_i64[0] != v94 )
LABEL_71:
    j_j___libc_free_0(v36);
LABEL_43:
  if ( v92.m128i_i32[2] > 0x40u )
  {
    v42 = v92.m128i_i64[0];
    if ( v92.m128i_i64[0] )
LABEL_45:
      j_j___libc_free_0_0(v42);
  }
  return a1;
}
