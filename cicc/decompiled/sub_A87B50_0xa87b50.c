// Function: sub_A87B50
// Address: 0xa87b50
//
__int64 *__fastcall sub_A87B50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __m128i *v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  _BYTE *v12; // r15
  _BYTE *v13; // rdi
  size_t v14; // r14
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rdx
  char *v27; // rsi
  __m128i *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  const char *v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  _BYTE *v35; // rdi
  __int64 v36; // r8
  __int64 v37; // rdx
  __m128i *v38; // rsi
  _BYTE *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdi
  __m128i v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r8
  unsigned __int64 v45; // r9
  unsigned __int64 v46; // r10
  const char *v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r8
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r9
  __int64 v57; // rdx
  void *v58; // [rsp+18h] [rbp-248h]
  __m128i src; // [rsp+20h] [rbp-240h] BYREF
  __m128i v60; // [rsp+30h] [rbp-230h] BYREF
  _BYTE v61[16]; // [rsp+40h] [rbp-220h] BYREF
  __m128i dest; // [rsp+50h] [rbp-210h] BYREF
  __m128i v63; // [rsp+60h] [rbp-200h] BYREF
  _QWORD v64[2]; // [rsp+70h] [rbp-1F0h] BYREF
  char v65[16]; // [rsp+80h] [rbp-1E0h] BYREF
  _BYTE v66[32]; // [rsp+90h] [rbp-1D0h] BYREF
  __m128i v67; // [rsp+B0h] [rbp-1B0h] BYREF
  __int16 v68; // [rsp+D0h] [rbp-190h]
  __m128i v69[2]; // [rsp+E0h] [rbp-180h] BYREF
  __int16 v70; // [rsp+100h] [rbp-160h]
  __m128i v71; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v72[4]; // [rsp+120h] [rbp-140h] BYREF
  __m128i v73; // [rsp+140h] [rbp-120h] BYREF
  const char *v74; // [rsp+150h] [rbp-110h]
  __int64 v75; // [rsp+158h] [rbp-108h]
  __int16 v76; // [rsp+160h] [rbp-100h]
  __m128i v77; // [rsp+170h] [rbp-F0h] BYREF
  const char *v78; // [rsp+180h] [rbp-E0h] BYREF
  __int64 v79; // [rsp+188h] [rbp-D8h]
  __int16 v80; // [rsp+190h] [rbp-D0h]
  _QWORD v81[2]; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+1B0h] [rbp-B0h] BYREF
  int v83; // [rsp+1C0h] [rbp-A0h]
  int v84; // [rsp+1CCh] [rbp-94h]
  int v85; // [rsp+1D0h] [rbp-90h]
  __m128i v86; // [rsp+1E0h] [rbp-80h] BYREF
  _QWORD v87[2]; // [rsp+1F0h] [rbp-70h] BYREF
  __int16 v88; // [rsp+200h] [rbp-60h]

  src.m128i_i64[0] = a2;
  src.m128i_i64[1] = a3;
  v88 = 261;
  v86.m128i_i64[0] = a4;
  v86.m128i_i64[1] = a5;
  sub_CC9F70(v81, &v86);
  v6 = v83;
  if ( v83 == 26 || (unsigned int)(v83 - 48) <= 1 || (unsigned int)(v83 - 51) <= 1 )
  {
    if ( sub_C931B0(&src, "-G", 2, 0) == -1 )
    {
      if ( !src.m128i_i64[1] )
      {
        *a1 = (__int64)(a1 + 2);
        sub_A7BD10(a1, "G1", (__int64)"");
        goto LABEL_22;
      }
      if ( *(_BYTE *)src.m128i_i64[0] != 71 )
      {
        v88 = 773;
        v86 = src;
        v87[0] = "-G1";
        sub_CA0F50(a1, &v86);
        goto LABEL_22;
      }
    }
    v6 = v83;
  }
  if ( v6 != 29 && v6 != 14 )
  {
    if ( src.m128i_i64[0] )
    {
      dest.m128i_i64[0] = (__int64)&v63;
      sub_A7BD10(dest.m128i_i64, src.m128i_i64[0], src.m128i_i64[0] + src.m128i_i64[1]);
      v6 = v83;
    }
    else
    {
      v63.m128i_i8[0] = 0;
      dest.m128i_i64[0] = (__int64)&v63;
      dest.m128i_i64[1] = 0;
    }
    if ( v6 != 27 )
    {
      if ( (unsigned int)(v6 - 3) <= 2 )
      {
        if ( !src.m128i_i64[1] || sub_C931B0(&src, "-Fn32", 5, 0) != -1 )
        {
LABEL_12:
          if ( sub_C931B0(&src, "-p270:32:32-p271:32:32-p272:64:64", 33, 0) != -1 )
            goto LABEL_13;
          v86.m128i_i64[1] = 0x400000000LL;
          v86.m128i_i64[0] = (__int64)v87;
          sub_C88F40(v69, "^([Ee]-m:[a-z](-p:32:32)?)(-.*)$", 32, 0);
          v28 = (__m128i *)dest.m128i_i64[0];
          if ( !(unsigned __int8)sub_C89090(v69, dest.m128i_i64[0], dest.m128i_i64[1], &v86, 0) )
          {
LABEL_95:
            sub_C88FF0(v69);
            if ( (_QWORD *)v86.m128i_i64[0] != v87 )
              _libc_free(v86.m128i_i64[0], v28);
LABEL_13:
            *a1 = (__int64)(a1 + 2);
            v7 = (__m128i *)dest.m128i_i64[0];
            if ( (__m128i *)dest.m128i_i64[0] != &v63 )
            {
LABEL_14:
              *a1 = (__int64)v7;
              a1[2] = v63.m128i_i64[0];
LABEL_15:
              a1[1] = dest.m128i_i64[1];
              goto LABEL_22;
            }
LABEL_131:
            *((__m128i *)a1 + 1) = _mm_load_si128(&v63);
            goto LABEL_15;
          }
          v31 = *(const char **)(v86.m128i_i64[0] + 48);
          v32 = *(_QWORD *)(v86.m128i_i64[0] + 56);
          v33 = *(_QWORD *)(v86.m128i_i64[0] + 16);
          v34 = *(_QWORD *)(v86.m128i_i64[0] + 24);
          v78 = "-p270:32:32-p271:32:32-p272:64:64";
          v75 = v32;
          v77.m128i_i64[1] = v34;
          v77.m128i_i64[0] = v33;
          v28 = &v73;
          v80 = 1285;
          v73.m128i_i64[0] = (__int64)&v77;
          v74 = v31;
          v79 = 33;
          v76 = 1282;
          sub_CA0F50(&v71, &v73);
          v35 = (_BYTE *)dest.m128i_i64[0];
          if ( (_QWORD *)v71.m128i_i64[0] == v72 )
          {
            v37 = v71.m128i_i64[1];
            if ( v71.m128i_i64[1] )
            {
              if ( v71.m128i_i64[1] == 1 )
              {
                *(_BYTE *)dest.m128i_i64[0] = v72[0];
              }
              else
              {
                v28 = (__m128i *)v72;
                memcpy((void *)dest.m128i_i64[0], v72, v71.m128i_u64[1]);
              }
              v37 = v71.m128i_i64[1];
              v35 = (_BYTE *)dest.m128i_i64[0];
            }
            dest.m128i_i64[1] = v37;
            v35[v37] = 0;
            v35 = (_BYTE *)v71.m128i_i64[0];
            goto LABEL_120;
          }
          v28 = (__m128i *)v71.m128i_i64[1];
          if ( (__m128i *)dest.m128i_i64[0] == &v63 )
          {
            dest = v71;
            v63.m128i_i64[0] = v72[0];
          }
          else
          {
            v36 = v63.m128i_i64[0];
            dest = v71;
            v63.m128i_i64[0] = v72[0];
            if ( v35 )
            {
              v71.m128i_i64[0] = (__int64)v35;
              v72[0] = v36;
              goto LABEL_120;
            }
          }
          v71.m128i_i64[0] = (__int64)v72;
          v35 = v72;
LABEL_120:
          v71.m128i_i64[1] = 0;
          *v35 = 0;
          if ( (_QWORD *)v71.m128i_i64[0] != v72 )
          {
            v28 = (__m128i *)(v72[0] + 1LL);
            j_j___libc_free_0(v71.m128i_i64[0], v72[0] + 1LL);
          }
          goto LABEL_95;
        }
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) > 4 )
        {
          sub_2241490(&dest, "-Fn32", 5, v25);
          goto LABEL_12;
        }
LABEL_151:
        sub_4262D8((__int64)"basic_string::append");
      }
      if ( (unsigned int)(v6 - 30) <= 2 )
        goto LABEL_40;
      if ( (unsigned int)(v6 - 18) <= 1 )
      {
        if ( sub_C931B0(&src, "m:m", 3, 0) == -1 )
          goto LABEL_40;
        v6 = v83;
      }
      if ( (v6 & 0xFFFFFFDE) != 0x18 )
      {
        v58 = a1 + 2;
        if ( (unsigned int)(v6 - 38) > 1 )
        {
          v7 = (__m128i *)dest.m128i_i64[0];
          *a1 = (__int64)(a1 + 2);
          if ( v7 != &v63 )
            goto LABEL_14;
          goto LABEL_131;
        }
        if ( sub_C931B0(&src, "-p270:32:32-p271:32:32-p272:64:64", 33, 0) != -1 )
        {
LABEL_104:
          if ( v84 != 26 )
          {
            strcpy(v65, "-i128:128");
            v64[0] = v65;
            v60 = dest;
            v64[1] = 9;
            if ( sub_C931B0(&v60, v65, 9, 0) == -1 )
            {
              v86.m128i_i64[0] = (__int64)v87;
              v86.m128i_i64[1] = 0x400000000LL;
              sub_C88F40(v61, "^(e(-[mpi][^-]*)*)((-[^mpi][^-]*)*)$", 36, 0);
              v39 = (_BYTE *)dest.m128i_i64[0];
              if ( (unsigned __int8)sub_C89090(v61, dest.m128i_i64[0], dest.m128i_i64[1], &v86, 0) )
              {
                v76 = 261;
                v68 = 261;
                v73.m128i_i64[0] = *(_QWORD *)(v86.m128i_i64[0] + 48);
                v53 = *(_QWORD *)(v86.m128i_i64[0] + 56);
                v69[0].m128i_i64[0] = (__int64)v64;
                v73.m128i_i64[1] = v53;
                v70 = 260;
                v54 = *(_QWORD *)(v86.m128i_i64[0] + 24);
                v67.m128i_i64[0] = *(_QWORD *)(v86.m128i_i64[0] + 16);
                v67.m128i_i64[1] = v54;
                sub_9C6370(&v71, &v67, v69, (__int64)v64, (__int64)&v71, (__int64)v61);
                sub_9C6370(&v77, &v71, &v73, v55, (__int64)&v71, v56);
                sub_CA0F50(v66, &v77);
                v39 = v66;
                sub_A7BC30((__int64)&dest, (__int64)v66);
                sub_2240A30(v66);
              }
              sub_C88FF0(v61);
              if ( (_QWORD *)v86.m128i_i64[0] != v87 )
                _libc_free(v86.m128i_i64[0], v39);
            }
            sub_2240A30(v64);
            if ( v84 == 14 && (v85 == 27 || !v85) && !(unsigned __int8)sub_CC7F40(v81) )
            {
              v69[0] = dest;
              v40 = sub_C931B0(v69, "-f80:32-", 8, 0);
              if ( v40 != -1 )
              {
                v41 = v69[0].m128i_i64[0];
                v42.m128i_i64[0] = sub_A7BB90(v69[0].m128i_i64[0], v69[0].m128i_u64[1], v40 + 8);
                v73.m128i_i64[0] = v41;
                if ( v45 > v46 )
                  v45 = v46;
                v77 = v42;
                v73.m128i_i64[1] = v45;
                v74 = "-f80:128-";
                v80 = 261;
                v76 = 773;
                sub_9C6370(&v86, &v73, &v77, v43, v44, v45);
                sub_CA0F50(&v71, &v86);
                sub_A7BC30((__int64)&dest, (__int64)&v71);
                sub_2240A30(&v71);
              }
            }
          }
          *a1 = (__int64)v58;
          v7 = (__m128i *)dest.m128i_i64[0];
          if ( (__m128i *)dest.m128i_i64[0] != &v63 )
            goto LABEL_14;
          goto LABEL_131;
        }
        v86.m128i_i64[0] = (__int64)v87;
        v86.m128i_i64[1] = 0x400000000LL;
        sub_C88F40(v69, "^([Ee]-m:[a-z](-p:32:32)?)(-.*)$", 32, 0);
        v38 = (__m128i *)dest.m128i_i64[0];
        if ( !(unsigned __int8)sub_C89090(v69, dest.m128i_i64[0], dest.m128i_i64[1], &v86, 0) )
        {
LABEL_134:
          sub_C88FF0(v69);
          if ( (_QWORD *)v86.m128i_i64[0] != v87 )
            _libc_free(v86.m128i_i64[0], v38);
          goto LABEL_104;
        }
        v47 = *(const char **)(v86.m128i_i64[0] + 48);
        v48 = *(_QWORD *)(v86.m128i_i64[0] + 16);
        v49 = *(_QWORD *)(v86.m128i_i64[0] + 56);
        v50 = *(_QWORD *)(v86.m128i_i64[0] + 24);
        v79 = 33;
        v77.m128i_i64[0] = v48;
        v38 = &v73;
        v77.m128i_i64[1] = v50;
        v78 = "-p270:32:32-p271:32:32-p272:64:64";
        v73.m128i_i64[0] = (__int64)&v77;
        v74 = v47;
        v80 = 1285;
        v75 = v49;
        v76 = 1282;
        sub_CA0F50(&v71, &v73);
        v51 = dest.m128i_i64[0];
        if ( (_QWORD *)v71.m128i_i64[0] == v72 )
        {
          v57 = v71.m128i_i64[1];
          if ( v71.m128i_i64[1] )
          {
            if ( v71.m128i_i64[1] == 1 )
            {
              *(_BYTE *)dest.m128i_i64[0] = v72[0];
            }
            else
            {
              v38 = (__m128i *)v72;
              memcpy((void *)dest.m128i_i64[0], v72, v71.m128i_u64[1]);
            }
            v51 = dest.m128i_i64[0];
            v57 = v71.m128i_i64[1];
          }
          dest.m128i_i64[1] = v57;
          *(_BYTE *)(v51 + v57) = 0;
          goto LABEL_148;
        }
        v38 = (__m128i *)v71.m128i_i64[1];
        if ( (__m128i *)dest.m128i_i64[0] == &v63 )
        {
          dest = v71;
          v63.m128i_i64[0] = v72[0];
        }
        else
        {
          v52 = v63.m128i_i64[0];
          dest = v71;
          v63.m128i_i64[0] = v72[0];
          if ( v51 )
          {
            v71.m128i_i64[0] = v51;
            v72[0] = v52;
            goto LABEL_148;
          }
        }
        v71.m128i_i64[0] = (__int64)v72;
LABEL_148:
        v71.m128i_i64[1] = 0;
        *(_BYTE *)v71.m128i_i64[0] = 0;
        if ( (_QWORD *)v71.m128i_i64[0] != v72 )
        {
          v38 = (__m128i *)(v72[0] + 1LL);
          j_j___libc_free_0(v71.m128i_i64[0], v72[0] + 1LL);
        }
        goto LABEL_134;
      }
LABEL_40:
      v77.m128i_i64[0] = (__int64)&v78;
      sub_A7BD10(v77.m128i_i64, "-i64:64", (__int64)"");
      v86.m128i_i64[0] = (__int64)v87;
      sub_A7BD10(v86.m128i_i64, "-i128:128", (__int64)"");
      v73 = dest;
      if ( sub_C931B0(&v73, v86.m128i_i64[0], v86.m128i_i64[1], 0) == -1 )
      {
        v30 = sub_22416F0(&dest, v77.m128i_i64[0], 0, v77.m128i_i64[1]);
        if ( v30 != -1 )
        {
          if ( (unsigned __int64)(v77.m128i_i64[1] + v30) > dest.m128i_i64[1] )
            sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::replace");
          sub_2241130(&dest, v77.m128i_i64[1] + v30, 0, v86.m128i_i64[0], v86.m128i_i64[1]);
        }
      }
      *a1 = (__int64)(a1 + 2);
      if ( (__m128i *)dest.m128i_i64[0] == &v63 )
      {
        *((__m128i *)a1 + 1) = _mm_load_si128(&v63);
      }
      else
      {
        *a1 = dest.m128i_i64[0];
        a1[2] = v63.m128i_i64[0];
      }
      v15 = dest.m128i_i64[1];
      v16 = (_QWORD *)v86.m128i_i64[0];
      dest.m128i_i64[0] = (__int64)&v63;
      dest.m128i_i64[1] = 0;
      a1[1] = v15;
      v63.m128i_i8[0] = 0;
      if ( v16 != v87 )
        j_j___libc_free_0(v16, v87[0] + 1LL);
      if ( (const char **)v77.m128i_i64[0] != &v78 )
        j_j___libc_free_0(v77.m128i_i64[0], v78 + 1);
      if ( (__m128i *)dest.m128i_i64[0] != &v63 )
        j_j___libc_free_0(dest.m128i_i64[0], v63.m128i_i64[0] + 1);
      goto LABEL_22;
    }
    if ( sub_C931B0(&src, "-G", 2, 0) == -1 && (!src.m128i_i64[1] || *(_BYTE *)src.m128i_i64[0] != 71) )
    {
      if ( dest.m128i_i64[1] )
      {
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 2 )
          goto LABEL_151;
        v26 = 3;
        v27 = "-G1";
      }
      else
      {
        v26 = 2;
        v27 = "G1";
      }
      sub_2241490(&dest, v27, v26, v17);
    }
    if ( sub_C931B0(&src, "-ni", 3, 0) == -1 )
    {
      v19 = src.m128i_u64[1];
      if ( src.m128i_i64[1] > 1uLL )
      {
        v20 = src.m128i_i64[0];
        if ( *(_WORD *)src.m128i_i64[0] == 26990 )
          goto LABEL_52;
      }
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 8 )
        goto LABEL_151;
      sub_2241490(&dest, "-ni:7:8:9", 9, v18);
    }
    v19 = src.m128i_u64[1];
    v20 = src.m128i_i64[0];
LABEL_52:
    if ( v19 > 3 )
    {
      if ( *(_DWORD *)(v20 + v19 - 4) == 926574958 )
      {
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 3 )
          goto LABEL_151;
        sub_2241490(&dest, ":8:9", 4, v18);
        v19 = src.m128i_u64[1];
        v20 = src.m128i_i64[0];
      }
      if ( v19 > 5 )
      {
        v21 = v20 + v19 - 6;
        if ( *(_DWORD *)v21 == 926574958 && *(_WORD *)(v21 + 4) == 14394 )
        {
          if ( dest.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL || dest.m128i_i64[1] == 4611686018427387902LL )
            goto LABEL_151;
          sub_2241490(&dest, ":9", 2, v18);
        }
      }
    }
    if ( sub_C931B0(&src, "-p7", 3, 0) == -1 && (src.m128i_i64[1] <= 1uLL || *(_WORD *)src.m128i_i64[0] != 14192) )
    {
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 0x11 )
        goto LABEL_151;
      sub_2241490(&dest, "-p7:160:256:256:32", 18, v22);
    }
    if ( sub_C931B0(&src, "-p8", 3, 0) == -1 && (src.m128i_i64[1] <= 1uLL || *(_WORD *)src.m128i_i64[0] != 14448) )
    {
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 0xA )
        goto LABEL_151;
      sub_2241490(&dest, "-p8:128:128", 11, v23);
    }
    if ( sub_C931B0(&src, "-p9", 3, 0) == -1 && (src.m128i_i64[1] <= 1uLL || *(_WORD *)src.m128i_i64[0] != 14704) )
    {
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - dest.m128i_i64[1]) <= 0x11 )
        goto LABEL_151;
      sub_2241490(&dest, "-p9:192:256:256:32", 18, v24);
    }
    *a1 = (__int64)(a1 + 2);
    v7 = (__m128i *)dest.m128i_i64[0];
    if ( (__m128i *)dest.m128i_i64[0] != &v63 )
      goto LABEL_14;
    goto LABEL_131;
  }
  v8 = sub_C931B0(&src, "-n64-", 5, 0);
  if ( v8 == -1 )
  {
    v12 = (_BYTE *)src.m128i_i64[0];
    v13 = a1 + 2;
    if ( !src.m128i_i64[0] )
    {
      *a1 = (__int64)v13;
      a1[1] = 0;
      *((_BYTE *)a1 + 16) = 0;
      goto LABEL_22;
    }
    v14 = src.m128i_u64[1];
    *a1 = (__int64)v13;
    v86.m128i_i64[0] = v14;
    if ( v14 > 0xF )
    {
      v29 = sub_22409D0(a1, &v86, 0);
      *a1 = v29;
      v13 = (_BYTE *)v29;
      a1[2] = v86.m128i_i64[0];
    }
    else
    {
      if ( v14 == 1 )
      {
        *((_BYTE *)a1 + 16) = *v12;
LABEL_34:
        a1[1] = v14;
        v13[v14] = 0;
        goto LABEL_22;
      }
      if ( !v14 )
        goto LABEL_34;
    }
    memcpy(v13, v12, v14);
    v14 = v86.m128i_i64[0];
    v13 = (_BYTE *)*a1;
    goto LABEL_34;
  }
  v9 = v8 + 5;
  if ( v8 + 5 > src.m128i_i64[1] )
  {
    v9 = src.m128i_u64[1];
    v10 = 0;
  }
  else
  {
    v10 = src.m128i_i64[1] - v9;
  }
  v87[1] = v10;
  if ( v8 > src.m128i_i64[1] )
    v8 = src.m128i_u64[1];
  v77.m128i_i64[0] = src.m128i_i64[0];
  v80 = 773;
  v77.m128i_i64[1] = v8;
  v78 = "-n32:64-";
  v86.m128i_i64[0] = (__int64)&v77;
  v87[0] = v9 + src.m128i_i64[0];
  v88 = 1282;
  sub_CA0F50(a1, &v86);
LABEL_22:
  if ( (__int64 *)v81[0] != &v82 )
    j_j___libc_free_0(v81[0], v82 + 1);
  return a1;
}
