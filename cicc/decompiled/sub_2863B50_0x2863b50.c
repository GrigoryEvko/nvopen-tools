// Function: sub_2863B50
// Address: 0x2863b50
//
void __fastcall sub_2863B50(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, int a5, __int64 a6, char a7)
{
  __int64 v11; // r12
  __int64 v12; // rcx
  __int64 *v13; // r8
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 *v20; // r12
  __int64 v21; // r8
  char *v22; // r14
  __int64 *v23; // r8
  int v24; // edx
  __int64 *v25; // rdi
  size_t v26; // r9
  unsigned __int64 v27; // r14
  __int64 v28; // rdx
  __int64 *v29; // rax
  size_t v30; // r8
  __int64 v31; // r14
  _QWORD *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  __m128i v36; // xmm0
  __int64 v37; // rcx
  char v38; // al
  __m128i v39; // xmm1
  __int64 v40; // rax
  __int64 v41; // r8
  char v42; // al
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  unsigned int v49; // eax
  unsigned int v50; // edx
  __int64 v51; // r8
  __int64 v52; // rax
  __m128i v53; // xmm2
  char v54; // dl
  __int64 v55; // rdx
  __int64 v56; // rdx
  __m128i v57; // xmm3
  __int64 v58; // rcx
  __int64 v59; // rcx
  _QWORD *v60; // rdx
  _BYTE *v61; // rsi
  _BYTE *v62; // rdx
  int v63; // eax
  __int64 v64; // [rsp+8h] [rbp-1F8h]
  int v65; // [rsp+10h] [rbp-1F0h]
  __int64 v67; // [rsp+20h] [rbp-1E0h]
  char *v68; // [rsp+20h] [rbp-1E0h]
  size_t v69; // [rsp+20h] [rbp-1E0h]
  __int64 v70; // [rsp+20h] [rbp-1E0h]
  __int64 v71; // [rsp+20h] [rbp-1E0h]
  __int64 *v73; // [rsp+30h] [rbp-1D0h]
  __int64 *v74; // [rsp+30h] [rbp-1D0h]
  __int64 *v75; // [rsp+40h] [rbp-1C0h]
  void *src; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 v77; // [rsp+58h] [rbp-1A8h]
  _BYTE v78[64]; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 *v79; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-158h]
  _BYTE v81[64]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v82; // [rsp+F0h] [rbp-110h] BYREF
  __m128i v83; // [rsp+F8h] [rbp-108h]
  char v84; // [rsp+108h] [rbp-F8h]
  __int64 v85; // [rsp+110h] [rbp-F0h]
  _BYTE *v86; // [rsp+118h] [rbp-E8h] BYREF
  __int64 v87; // [rsp+120h] [rbp-E0h]
  _BYTE v88[32]; // [rsp+128h] [rbp-D8h] BYREF
  _QWORD *v89; // [rsp+148h] [rbp-B8h]
  __m128i v90; // [rsp+150h] [rbp-B0h]
  __int64 v91; // [rsp+160h] [rbp-A0h] BYREF
  __m128i v92; // [rsp+168h] [rbp-98h]
  char v93; // [rsp+178h] [rbp-88h]
  __int64 v94; // [rsp+180h] [rbp-80h]
  unsigned __int64 v95[2]; // [rsp+188h] [rbp-78h] BYREF
  _BYTE v96[32]; // [rsp+198h] [rbp-68h] BYREF
  __int64 v97; // [rsp+1B8h] [rbp-48h]
  __m128i v98; // [rsp+1C0h] [rbp-40h]

  if ( !a7 )
  {
    v11 = *(_QWORD *)(a4[5] + 8 * a6);
    if ( *(_DWORD *)(a1 + 72) != 1 )
      goto LABEL_3;
LABEL_62:
    if ( sub_285BA20(*(_QWORD *)(a1 + 48), a2, v11, *(_QWORD *)(a1 + 56), *(_QWORD *)(a1 + 8)) )
      return;
    goto LABEL_3;
  }
  v11 = a4[11];
  if ( *(_DWORD *)(a1 + 72) == 1 )
    goto LABEL_62;
LABEL_3:
  v12 = *(_QWORD *)(a1 + 56);
  v13 = *(__int64 **)(a1 + 8);
  src = v78;
  v77 = 0x800000000LL;
  v16 = sub_285D500(v11, 0, (__int64)&src, v12, v13, 0);
  if ( v16 )
  {
    v17 = (unsigned int)v77;
    v18 = (unsigned int)v77 + 1LL;
    if ( v18 > HIDWORD(v77) )
    {
      sub_C8D5F0((__int64)&src, v78, v18, 8u, v14, v15);
      v17 = (unsigned int)v77;
    }
    *((_QWORD *)src + v17) = v16;
    v19 = v77 + 1;
    LODWORD(v77) = v77 + 1;
  }
  else
  {
    v19 = v77;
  }
  v20 = (__int64 *)src;
  if ( v19 != 1 )
  {
    v75 = (__int64 *)((char *)src + 8 * v19);
    if ( v75 != src )
    {
      v64 = 8 * a6;
      while ( 1 )
      {
        v21 = *v20;
        v22 = (char *)v20++;
        if ( *(_WORD *)(v21 + 24) == 15 )
        {
          if ( !sub_DADE90(*(_QWORD *)(a1 + 8), v21, *(_QWORD *)(a1 + 56)) )
            goto LABEL_51;
          v21 = *(v20 - 1);
        }
        if ( !sub_2857DF0(
                *(__int64 **)(a1 + 48),
                *(__int64 **)(a1 + 8),
                *(_QWORD *)(a2 + 712),
                *(_BYTE *)(a2 + 720),
                *(_QWORD *)(a2 + 728),
                *(_BYTE *)(a2 + 736),
                *(_DWORD *)(a2 + 32),
                *(_QWORD *)(a2 + 40),
                *(_DWORD *)(a2 + 48),
                v21,
                *((unsigned int *)a4 + 12) - ((unsigned __int64)(a4[11] == 0) - 1) > 1) )
        {
          v23 = (__int64 *)src;
          v24 = 0;
          v25 = (__int64 *)v81;
          v26 = v22 - (_BYTE *)src;
          v79 = (__int64 *)v81;
          v80 = 0x800000000LL;
          v27 = (v22 - (_BYTE *)src) >> 3;
          if ( v26 > 0x40 )
          {
            v69 = v26;
            v74 = (__int64 *)src;
            sub_C8D5F0((__int64)&v79, v81, v27, 8u, (__int64)src, v26);
            v24 = v80;
            v26 = v69;
            v23 = v74;
            v25 = &v79[(unsigned int)v80];
          }
          if ( v23 != v20 - 1 )
          {
            memcpy(v25, v23, v26);
            v24 = v80;
          }
          LODWORD(v80) = v27 + v24;
          v28 = (unsigned int)(v27 + v24);
          v29 = (__int64 *)((char *)src + 8 * (unsigned int)v77);
          v30 = (char *)v29 - (char *)v20;
          v31 = v29 - v20;
          if ( v28 + v31 > (unsigned __int64)HIDWORD(v80) )
          {
            v68 = (char *)((_BYTE *)src + 8 * (unsigned int)v77 - (_BYTE *)v20);
            v73 = (__int64 *)((char *)src + 8 * (unsigned int)v77);
            sub_C8D5F0((__int64)&v79, v81, v28 + v31, 8u, v30, v28 + v31);
            v28 = (unsigned int)v80;
            v30 = (size_t)v68;
            v29 = v73;
          }
          if ( v29 != v20 )
          {
            memcpy(&v79[v28], v20, v30);
            LODWORD(v28) = v80;
          }
          LODWORD(v80) = v31 + v28;
          if ( (_DWORD)v31 + (_DWORD)v28 != 1
            || !sub_2857DF0(
                  *(__int64 **)(a1 + 48),
                  *(__int64 **)(a1 + 8),
                  *(_QWORD *)(a2 + 712),
                  *(_BYTE *)(a2 + 720),
                  *(_QWORD *)(a2 + 728),
                  *(_BYTE *)(a2 + 736),
                  *(_DWORD *)(a2 + 32),
                  *(_QWORD *)(a2 + 40),
                  *(_DWORD *)(a2 + 48),
                  *v79,
                  *((unsigned int *)a4 + 12) - ((unsigned __int64)(a4[11] == 0) - 1) > 1) )
          {
            v32 = sub_DC7EB0(*(__int64 **)(a1 + 8), (__int64)&v79, 0, 0);
            if ( !sub_D968A0((__int64)v32) )
            {
              v36 = _mm_loadu_si128((const __m128i *)(a4 + 1));
              v37 = *((unsigned int *)a4 + 12);
              v82 = *a4;
              v38 = *((_BYTE *)a4 + 24);
              v83 = v36;
              v84 = v38;
              v85 = a4[4];
              v86 = v88;
              v87 = 0x400000000LL;
              if ( (_DWORD)v37 )
                sub_2850210((__int64)&v86, (__int64)(a4 + 5), v33, v37, v34, v35);
              v39 = _mm_loadu_si128((const __m128i *)a4 + 6);
              v89 = (_QWORD *)a4[11];
              v40 = a4[12];
              v90 = v39;
              if ( !v40 || !v90.m128i_i8[8] )
              {
                if ( !*((_WORD *)v32 + 12)
                  && (unsigned __int64)sub_D97050(*(_QWORD *)(a1 + 8), *(_QWORD *)(v32[4] + 8LL)) <= 0x40
                  && (unsigned __int8)sub_DFA0C0(*(_QWORD *)(a1 + 48)) )
                {
                  v59 = v32[4];
                  v60 = *(_QWORD **)(v59 + 24);
                  if ( *(_DWORD *)(v59 + 32) > 0x40u )
                    v60 = (_QWORD *)*v60;
                  v90.m128i_i8[8] = 0;
                  v90.m128i_i64[0] += (__int64)v60;
                  if ( a7 )
                  {
                    v89 = 0;
                    v85 = 0;
                  }
                  else
                  {
                    v61 = &v86[v64 + 8];
                    v62 = &v86[8 * (unsigned int)v87];
                    v63 = v87;
                    if ( v62 != v61 )
                    {
                      memmove(&v86[v64], v61, v62 - v61);
                      v63 = v87;
                    }
                    LODWORD(v87) = v63 - 1;
                  }
                }
                else if ( a7 )
                {
                  v89 = v32;
                }
                else
                {
                  *(_QWORD *)&v86[v64] = v32;
                }
                v41 = *(v20 - 1);
                if ( *(_WORD *)(v41 + 24) )
                {
LABEL_35:
                  v43 = (unsigned int)v87;
                  v44 = (unsigned int)v87 + 1LL;
                  if ( v44 > HIDWORD(v87) )
                  {
                    v71 = v41;
                    sub_C8D5F0((__int64)&v86, v88, v44, 8u, v41, v35);
                    v43 = (unsigned int)v87;
                    v41 = v71;
                  }
                  v45 = (__int64)v86;
                  *(_QWORD *)&v86[8 * v43] = v41;
                  LODWORD(v87) = v87 + 1;
                }
                else
                {
                  v67 = *(v20 - 1);
                  if ( (unsigned __int64)sub_D97050(*(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)(v41 + 32) + 8LL)) > 0x40
                    || (v42 = sub_DFA0C0(*(_QWORD *)(a1 + 48)), v41 = v67, !v42) )
                  {
                    v41 = *(v20 - 1);
                    goto LABEL_35;
                  }
                  v58 = *(_QWORD *)(v67 + 32);
                  v45 = *(_QWORD *)(v58 + 24);
                  if ( *(_DWORD *)(v58 + 32) > 0x40u )
                    v45 = *(_QWORD *)v45;
                  v90.m128i_i8[8] = 0;
                  v90.m128i_i64[0] += v45;
                }
                sub_2857080((__int64)&v82, *(_QWORD *)(a1 + 56), v45, (__int64)&v82, v41, v35);
                if ( (unsigned __int8)sub_2862B30(a1, a2, a3, (unsigned __int64)&v82, v46, v47) )
                {
                  v49 = 0x3FFFFFFF;
                  if ( (_DWORD)v77 )
                  {
                    _BitScanReverse(&v50, v77);
                    v49 = (31 - (v50 ^ 0x1F)) >> 2;
                  }
                  v51 = v49 + a5 + 1;
                  v52 = *(_QWORD *)(a2 + 760) + 112LL * *(unsigned int *)(a2 + 768) - 112;
                  v53 = _mm_loadu_si128((const __m128i *)(v52 + 8));
                  v91 = *(_QWORD *)v52;
                  v54 = *(_BYTE *)(v52 + 24);
                  v92 = v53;
                  v93 = v54;
                  v55 = *(_QWORD *)(v52 + 32);
                  v95[1] = 0x400000000LL;
                  v94 = v55;
                  v56 = *(unsigned int *)(v52 + 48);
                  v95[0] = (unsigned __int64)v96;
                  if ( (_DWORD)v56 )
                  {
                    v65 = v51;
                    v70 = v52;
                    sub_2850210((__int64)v95, v52 + 40, v56, v48, v51, (__int64)v96);
                    LODWORD(v51) = v65;
                    v52 = v70;
                  }
                  v57 = _mm_loadu_si128((const __m128i *)(v52 + 96));
                  v97 = *(_QWORD *)(v52 + 88);
                  v98 = v57;
                  if ( (unsigned int)v51 <= 2 )
                    sub_2864410(a1, a2, a3, &v91);
                  if ( (_BYTE *)v95[0] != v96 )
                    _libc_free(v95[0]);
                }
              }
              if ( v86 != v88 )
                _libc_free((unsigned __int64)v86);
            }
          }
          if ( v79 != (__int64 *)v81 )
            _libc_free((unsigned __int64)v79);
        }
LABEL_51:
        if ( v75 == v20 )
        {
          v20 = (__int64 *)src;
          break;
        }
      }
    }
  }
  if ( v20 != (__int64 *)v78 )
    _libc_free((unsigned __int64)v20);
}
