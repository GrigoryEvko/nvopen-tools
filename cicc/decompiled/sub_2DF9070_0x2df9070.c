// Function: sub_2DF9070
// Address: 0x2df9070
//
__int64 *__fastcall sub_2DF9070(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __m128i *v5; // rdx
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rax
  const char *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _WORD *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int64 v19; // rdi
  _WORD *v20; // rdx
  __int64 v21; // r13
  _BYTE *v22; // rax
  _WORD *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int **v28; // r13
  unsigned int *v29; // r14
  unsigned int *v30; // r8
  __int64 v31; // r9
  __int64 v32; // r10
  unsigned int *v33; // rbx
  __int64 v34; // rax
  char *v35; // r12
  size_t v36; // rax
  __int64 v37; // rcx
  size_t v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // rsi
  unsigned __int64 v41; // rsi
  char v42; // al
  __int64 v43; // rdx
  __int64 v44; // rdx
  int v45; // eax
  int v46; // ecx
  __int64 v47; // rbx
  unsigned __int64 v48; // r14
  __int64 v49; // rdx
  _DWORD *v50; // rdx
  __int64 v51; // rdi
  __int64 v52; // rdi
  _BYTE *v53; // rax
  _BYTE *v54; // rax
  __m128i v55; // xmm0
  __int64 **v56; // rbx
  __int64 *result; // rax
  __int64 v58; // rax
  _WORD *v59; // rdx
  _WORD *v60; // rdx
  __int64 *v61; // r12
  unsigned int v62; // esi
  __int64 *v63; // [rsp+0h] [rbp-130h]
  __int64 v65; // [rsp+10h] [rbp-120h]
  __int64 v66; // [rsp+18h] [rbp-118h]
  __int64 *v67; // [rsp+38h] [rbp-F8h]
  __int64 **v68; // [rsp+38h] [rbp-F8h]
  __int64 v69; // [rsp+40h] [rbp-F0h] BYREF
  _BYTE *v70; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v71; // [rsp+50h] [rbp-E0h]
  _BYTE v72[72]; // [rsp+58h] [rbp-D8h] BYREF
  __int64 v73; // [rsp+A0h] [rbp-90h] BYREF
  char *v74; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v75; // [rsp+B0h] [rbp-80h]
  _BYTE v76[120]; // [rsp+B8h] [rbp-78h] BYREF

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x25u )
  {
    sub_CB6200(a2, "********** DEBUG VARIABLES **********\n", 0x26u);
    v5 = *(__m128i **)(a2 + 32);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_444FCF0);
    v3[2].m128i_i32[0] = 707406378;
    v3[2].m128i_i16[2] = 2602;
    *v3 = si128;
    v3[1] = _mm_load_si128((const __m128i *)&xmmword_444FD00);
    v5 = (__m128i *)(*(_QWORD *)(a2 + 32) + 38LL);
    *(_QWORD *)(a2 + 32) = v5;
  }
  v6 = *(__int64 **)(a1 + 1000);
  v63 = &v6[*(unsigned int *)(a1 + 1008)];
  if ( v6 != v63 )
  {
    v67 = *(__int64 **)(a1 + 1000);
    do
    {
      v7 = *v67;
      v8 = *(_QWORD *)(a1 + 120);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 1u )
      {
        sub_CB6200(a2, "!\"", 2u);
      }
      else
      {
        v5->m128i_i16[0] = 8737;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      v9 = sub_B10CD0(v7 + 32);
      v10 = *(const char **)v7;
      sub_2DF7410(a2, *(_QWORD *)v7, v9);
      v14 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 1u )
      {
        v10 = "\"\t";
        sub_CB6200(a2, "\"\t", 2u);
      }
      else
      {
        *v14 = 2338;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      v73 = v7 + 232;
      v75 = 0x400000000LL;
      v74 = v76;
      sub_2DF64A0(&v73, (__int64)v10, (__int64)v14, v11, v12, v13);
      v71 = 0x400000000LL;
      v69 = v73;
      v70 = v72;
      if ( (_DWORD)v75 )
        sub_2DF4EE0((__int64)&v70, &v74, v15, v16, v17, v18);
      if ( v74 != v76 )
        _libc_free((unsigned __int64)v74);
      v19 = (unsigned __int64)v70;
      if ( (_DWORD)v71 )
      {
        v66 = v8;
        v65 = v7;
        do
        {
          if ( *(_DWORD *)(v19 + 12) >= *(_DWORD *)(v19 + 8) )
            break;
          v20 = *(_WORD **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v20 <= 1u )
          {
            v21 = sub_CB6200(a2, (unsigned __int8 *)" [", 2u);
          }
          else
          {
            v21 = a2;
            *v20 = 23328;
            *(_QWORD *)(a2 + 32) += 2LL;
          }
          v73 = *(_QWORD *)sub_2DF4990((__int64)&v69);
          sub_2FAD600(&v73, v21);
          v22 = *(_BYTE **)(v21 + 32);
          if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 24) )
          {
            v21 = sub_CB5D20(v21, 59);
          }
          else
          {
            *(_QWORD *)(v21 + 32) = v22 + 1;
            *v22 = 59;
          }
          v73 = *(_QWORD *)(*(_QWORD *)&v70[16 * (unsigned int)v71 - 16]
                          + 16LL * *(unsigned int *)&v70[16 * (unsigned int)v71 - 4]
                          + 8);
          sub_2FAD600(&v73, v21);
          v23 = *(_WORD **)(v21 + 32);
          if ( *(_QWORD *)(v21 + 24) - (_QWORD)v23 <= 1u )
          {
            sub_CB6200(v21, "):", 2u);
          }
          else
          {
            *v23 = 14889;
            *(_QWORD *)(v21 + 32) += 2LL;
          }
          v24 = (__int64)&v70[16 * (unsigned int)v71 - 16];
          v25 = *(unsigned int *)(v24 + 12);
          v26 = *(_QWORD *)v24;
          v27 = 3 * v25;
          v28 = (unsigned int **)(v26 + 8 * v27 + 64);
          if ( (*(_BYTE *)(v26 + 8 * v27 + 72) & 0x3F) != 0
            && (LODWORD(v73) = -1,
                v29 = (unsigned int *)(*(_QWORD *)(v26 + 8 * v27 + 64) + 4LL * (*(_BYTE *)(v26 + 8 * v27 + 72) & 0x3F)),
                v29 == sub_2DF4D60(*(_DWORD **)(v26 + 8 * v27 + 64), (__int64)v29, (int *)&v73)) )
          {
            if ( v30 != v29 )
            {
              v33 = v30;
              while ( 1 )
              {
                v35 = ", ";
                if ( v30 == v33 )
                  v35 = " ";
                v36 = strlen(v35);
                v37 = *(_QWORD *)(a2 + 32);
                v38 = v36;
                if ( *(_QWORD *)(a2 + 24) - v37 < v36 )
                {
                  ++v33;
                  v34 = sub_CB6200(a2, (unsigned __int8 *)v35, v36);
                  sub_CB59D0(v34, *(v33 - 1));
                  if ( v29 == v33 )
                    goto LABEL_35;
                }
                else
                {
                  if ( (_DWORD)v36 )
                  {
                    v39 = 0;
                    do
                    {
                      v40 = v39++;
                      *(_BYTE *)(v37 + v40) = v35[v40];
                    }
                    while ( v39 < (unsigned int)v38 );
                  }
                  *(_QWORD *)(a2 + 32) += v38;
                  v41 = *v33++;
                  sub_CB59D0(a2, v41);
                  if ( v29 == v33 )
                  {
LABEL_35:
                    v32 = (__int64)v70;
                    v31 = (unsigned int)v71;
                    break;
                  }
                }
                v30 = *v28;
              }
            }
            v42 = *(_BYTE *)(*(_QWORD *)(v32 + 16 * v31 - 16) + 24LL * *(unsigned int *)(v32 + 16 * v31 - 16 + 12) + 72);
            if ( (v42 & 0x40) != 0 )
            {
              sub_904010(a2, " ind");
              v32 = (__int64)v70;
              v31 = (unsigned int)v71;
            }
            else if ( v42 < 0 )
            {
              sub_904010(a2, " list");
              v32 = (__int64)v70;
              v31 = (unsigned int)v71;
            }
          }
          else
          {
            v43 = *(_QWORD *)(a2 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v43) <= 5 )
            {
              sub_CB6200(a2, " undef", 6u);
              v32 = (__int64)v70;
            }
            else
            {
              *(_DWORD *)v43 = 1684960544;
              *(_WORD *)(v43 + 4) = 26213;
              v32 = (__int64)v70;
              *(_QWORD *)(a2 + 32) += 6LL;
            }
            v31 = (unsigned int)v71;
          }
          v44 = v32 + 16 * v31 - 16;
          v45 = *(_DWORD *)(v44 + 12) + 1;
          *(_DWORD *)(v44 + 12) = v45;
          v19 = (unsigned __int64)v70;
          v46 = v71;
          if ( v45 == *(_DWORD *)&v70[16 * (unsigned int)v71 - 8] )
          {
            v62 = *(_DWORD *)(v69 + 160);
            if ( v62 )
            {
              sub_F03D40((__int64 *)&v70, v62);
              v46 = v71;
              v19 = (unsigned __int64)v70;
            }
          }
        }
        while ( v46 );
        v8 = v66;
        v7 = v65;
      }
      if ( (_BYTE *)v19 != v72 )
        _libc_free(v19);
      v47 = *(unsigned int *)(v7 + 64);
      v48 = 0;
      if ( (_DWORD)v47 )
      {
        do
        {
          v50 = *(_DWORD **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v50 <= 3u )
          {
            v51 = sub_CB6200(a2, " Loc", 4u);
          }
          else
          {
            *v50 = 1668238368;
            v51 = a2;
            *(_QWORD *)(a2 + 32) += 4LL;
          }
          v52 = sub_CB59D0(v51, v48);
          v53 = *(_BYTE **)(v52 + 32);
          if ( (unsigned __int64)v53 < *(_QWORD *)(v52 + 24) )
          {
            *(_QWORD *)(v52 + 32) = v53 + 1;
            *v53 = 61;
          }
          else
          {
            sub_CB5D20(v52, 61);
          }
          v49 = 5 * v48++;
          sub_2EAF9A0(*(_QWORD *)(v7 + 56) + 8 * v49, a2, v8);
        }
        while ( v48 != v47 );
      }
      v54 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v54 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 10);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v54 + 1;
        *v54 = 10;
      }
      ++v67;
      v5 = *(__m128i **)(a2 + 32);
    }
    while ( v63 != v67 );
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 0x22u )
  {
    sub_CB6200(a2, "********** DEBUG LABELS **********\n", 0x23u);
  }
  else
  {
    v55 = _mm_load_si128((const __m128i *)&xmmword_444FCF0);
    v5[2].m128i_i8[2] = 10;
    v5[2].m128i_i16[0] = 10794;
    *v5 = v55;
    v5[1] = _mm_load_si128((const __m128i *)&xmmword_444FD10);
    *(_QWORD *)(a2 + 32) += 35LL;
  }
  v56 = *(__int64 ***)(a1 + 1080);
  result = &v73;
  if ( &v56[*(unsigned int *)(a1 + 1088)] != v56 )
  {
    v68 = &v56[*(unsigned int *)(a1 + 1088)];
    do
    {
      while ( 1 )
      {
        v60 = *(_WORD **)(a2 + 32);
        v61 = *v56;
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v60 > 1u )
        {
          *v60 = 8737;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        else
        {
          sub_CB6200(a2, "!\"", 2u);
        }
        v58 = sub_B10CD0((__int64)(v61 + 1));
        sub_2DF7410(a2, *v61, v58);
        v59 = *(_WORD **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)v59 <= 1u )
        {
          sub_CB6200(a2, "\"\t", 2u);
        }
        else
        {
          *v59 = 2338;
          *(_QWORD *)(a2 + 32) += 2LL;
        }
        v73 = v61[2];
        sub_2FAD600(&v73, a2);
        result = *(__int64 **)(a2 + 32);
        if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
          break;
        ++v56;
        *(_QWORD *)(a2 + 32) = (char *)result + 1;
        *(_BYTE *)result = 10;
        if ( v68 == v56 )
          return result;
      }
      ++v56;
      result = (__int64 *)sub_CB5D20(a2, 10);
    }
    while ( v68 != v56 );
  }
  return result;
}
