// Function: sub_EA0260
// Address: 0xea0260
//
__int64 __fastcall sub_EA0260(
        __int64 a1,
        _DWORD *a2,
        size_t a3,
        void *a4,
        size_t a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9,
        __int64 a10,
        const char **a11,
        __int64 a12,
        const char **a13,
        __int64 a14)
{
  char **v17; // rbx
  char **v18; // r13
  char **v20; // r13
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __m128i v23; // xmm0
  _QWORD *v24; // rbx
  char *v25; // rdi
  _BYTE *v26; // rax
  size_t v27; // rdx
  _BYTE *v28; // rax
  __int64 v29; // rax
  const char **v30; // rax
  char *v31; // rsi
  size_t v32; // rdx
  __m128i v33; // xmm2
  __int64 v34; // rax
  _QWORD *v35; // rdi
  _BYTE *v36; // rax
  _QWORD *v37; // rdi
  __m128i *v38; // rax
  __m128i v39; // xmm0
  __int64 v40; // r8
  char *v41; // rdx
  const char **v42; // rax
  __int64 v43; // r15
  _BYTE *v44; // rax
  void *v45; // rdi
  __m128i *v46; // r13
  __m128i v47; // xmm0
  __m128i *v48; // rax
  __m128i v49; // xmm0
  _QWORD *v50; // r8
  _BYTE *v51; // rax
  void *v52; // rdi
  __int64 v53; // r8
  __m128i *v54; // rax
  __m128i si128; // xmm0
  __int64 v56; // rax
  __m128i v57; // xmm0
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // [rsp+8h] [rbp-B8h]
  _QWORD *v62; // [rsp+8h] [rbp-B8h]
  char *srca; // [rsp+10h] [rbp-B0h]
  _QWORD *s2; // [rsp+20h] [rbp-A0h]
  char **v66; // [rsp+28h] [rbp-98h]
  char **v67; // [rsp+40h] [rbp-80h] BYREF
  char **v68; // [rsp+48h] [rbp-78h]
  __int64 v69; // [rsp+50h] [rbp-70h]
  __m128i v70; // [rsp+60h] [rbp-60h] BYREF
  __m128i v71; // [rsp+70h] [rbp-50h] BYREF
  __int64 v72; // [rsp+80h] [rbp-40h]

  sub_F064E0(&v67, a7, a8);
  if ( !a12 || !a14 )
  {
    v17 = v68;
    v18 = v67;
    *(_OWORD *)a1 = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_OWORD *)(a1 + 16) = 0;
    goto LABEL_4;
  }
  v72 = 0;
  v70 = 0;
  v71 = 0;
  if ( a3 == 4 )
  {
    if ( *a2 == 1886152040 )
    {
      sub_E9FB40(a9, a10, a13, a14);
      goto LABEL_14;
    }
LABEL_38:
    v30 = sub_E9F8B0(a2, a3, a11, a12);
    if ( v30 )
    {
      sub_E9F500((__int64)&v70, (__int64)(v30 + 1), (__int64)a13, a14);
    }
    else
    {
      v50 = sub_CB72A0();
      v51 = (_BYTE *)v50[4];
      if ( (_BYTE *)v50[3] == v51 )
      {
        v58 = sub_CB6200((__int64)v50, (unsigned __int8 *)"'", 1u);
        v52 = *(void **)(v58 + 32);
        v50 = (_QWORD *)v58;
      }
      else
      {
        *v51 = 39;
        v52 = (void *)(v50[4] + 1LL);
        v50[4] = v52;
      }
      if ( v50[3] - (_QWORD)v52 < a3 )
      {
        v53 = sub_CB6200((__int64)v50, (unsigned __int8 *)a2, a3);
        v54 = *(__m128i **)(v53 + 32);
      }
      else
      {
        v62 = v50;
        memcpy(v52, a2, a3);
        v53 = (__int64)v62;
        v54 = (__m128i *)(v62[4] + a3);
        v62[4] = v54;
      }
      if ( *(_QWORD *)(v53 + 24) - (_QWORD)v54 <= 0x2Eu )
      {
        v53 = sub_CB6200(v53, "' is not a recognized processor for this target", 0x2Fu);
        v56 = *(_QWORD *)(v53 + 32);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
        qmemcpy(&v54[2], "for this target", 15);
        *v54 = si128;
        v54[1] = _mm_load_si128((const __m128i *)&xmmword_3F82970);
        v56 = *(_QWORD *)(v53 + 32) + 47LL;
        *(_QWORD *)(v53 + 32) = v56;
      }
      if ( (unsigned __int64)(*(_QWORD *)(v53 + 24) - v56) <= 0x15 )
      {
        sub_CB6200(v53, (unsigned __int8 *)" (ignoring processor)\n", 0x16u);
      }
      else
      {
        v57 = _mm_load_si128((const __m128i *)&xmmword_3F82980);
        *(_DWORD *)(v56 + 16) = 1919906675;
        *(_WORD *)(v56 + 20) = 2601;
        *(__m128i *)v56 = v57;
        *(_QWORD *)(v53 + 32) += 22LL;
      }
    }
    goto LABEL_14;
  }
  if ( a3 )
    goto LABEL_38;
LABEL_14:
  if ( a5 )
  {
    v42 = sub_E9F8B0(a4, a5, a11, a12);
    if ( v42 )
    {
      sub_E9F500((__int64)&v70, (__int64)(v42 + 6), (__int64)a13, a14);
    }
    else if ( a5 != a3 || memcmp(a4, a2, a5) )
    {
      v43 = (__int64)sub_CB72A0();
      v44 = *(_BYTE **)(v43 + 32);
      if ( *(_BYTE **)(v43 + 24) == v44 )
      {
        v59 = sub_CB6200(v43, (unsigned __int8 *)"'", 1u);
        v45 = *(void **)(v59 + 32);
        v43 = v59;
      }
      else
      {
        *v44 = 39;
        v45 = (void *)(*(_QWORD *)(v43 + 32) + 1LL);
        *(_QWORD *)(v43 + 32) = v45;
      }
      if ( *(_QWORD *)(v43 + 24) - (_QWORD)v45 < a5 )
      {
        v60 = sub_CB6200(v43, (unsigned __int8 *)a4, a5);
        v46 = *(__m128i **)(v60 + 32);
        v43 = v60;
      }
      else
      {
        memcpy(v45, a4, a5);
        v46 = (__m128i *)(*(_QWORD *)(v43 + 32) + a5);
        *(_QWORD *)(v43 + 32) = v46;
      }
      if ( *(_QWORD *)(v43 + 24) - (_QWORD)v46 <= 0x28u )
      {
        v43 = sub_CB6200(v43, "' is not a recognized processor for this ", 0x29u);
        v48 = *(__m128i **)(v43 + 32);
      }
      else
      {
        v47 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
        v46[2].m128i_i8[8] = 32;
        v46[2].m128i_i64[0] = 0x7369687420726F66LL;
        *v46 = v47;
        v46[1] = _mm_load_si128((const __m128i *)&xmmword_3F82970);
        v48 = (__m128i *)(*(_QWORD *)(v43 + 32) + 41LL);
        *(_QWORD *)(v43 + 32) = v48;
      }
      if ( *(_QWORD *)(v43 + 24) - (_QWORD)v48 <= 0x1Bu )
      {
        sub_CB6200(v43, "target (ignoring processor)\n", 0x1Cu);
      }
      else
      {
        v49 = _mm_load_si128((const __m128i *)&xmmword_3F82990);
        qmemcpy(&v48[1], " processor)\n", 12);
        *v48 = v49;
        *(_QWORD *)(v43 + 32) += 28LL;
      }
    }
  }
  v20 = v67;
  v17 = v68;
  if ( v67 == v68 )
  {
    v18 = v68;
    goto LABEL_42;
  }
  s2 = &a9[2 * a10];
  do
  {
    while ( 1 )
    {
      if ( !(unsigned int)sub_2241AC0(v20, "+help") )
      {
        sub_E9FB40(a9, a10, a13, a14);
        goto LABEL_18;
      }
      if ( (unsigned int)sub_2241AC0(v20, "+cpuhelp") )
        break;
      if ( !byte_4F8A328 )
      {
        v21 = sub_CB72A0();
        v22 = (__m128i *)v21[4];
        if ( v21[3] - (_QWORD)v22 <= 0x20u )
        {
          sub_CB6200((__int64)v21, "Available CPUs for this target:\n\n", 0x21u);
        }
        else
        {
          v23 = _mm_load_si128((const __m128i *)&xmmword_3F82890);
          v22[2].m128i_i8[0] = 10;
          *v22 = v23;
          v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F828A0);
          v21[4] += 33LL;
        }
        if ( s2 != a9 )
        {
          v66 = v17;
          v24 = a9;
          while ( 1 )
          {
            if ( v24[1] == 12 && *(_QWORD *)*v24 == 0x616C2D656C707061LL && *(_DWORD *)(*v24 + 8LL) == 1953719668 )
              goto LABEL_32;
            v40 = (__int64)sub_CB72A0();
            v28 = *(_BYTE **)(v40 + 32);
            if ( *(_BYTE **)(v40 + 24) == v28 )
            {
              v29 = sub_CB6200(v40, (unsigned __int8 *)"\t", 1u);
              v25 = *(char **)(v29 + 32);
              v40 = v29;
            }
            else
            {
              *v28 = 9;
              v25 = (char *)(*(_QWORD *)(v40 + 32) + 1LL);
              *(_QWORD *)(v40 + 32) = v25;
            }
            v26 = *(_BYTE **)(v40 + 24);
            v27 = v24[1];
            if ( v26 - v25 < v27 )
            {
              v40 = sub_CB6200(v40, (unsigned __int8 *)*v24, v27);
              v26 = *(_BYTE **)(v40 + 24);
              v25 = *(char **)(v40 + 32);
            }
            else if ( v27 )
            {
              v61 = v40;
              srca = (char *)v24[1];
              memcpy(v25, (const void *)*v24, v27);
              v40 = v61;
              v41 = &srca[*(_QWORD *)(v61 + 32)];
              v26 = *(_BYTE **)(v61 + 24);
              *(_QWORD *)(v61 + 32) = v41;
              v25 = v41;
            }
            if ( v26 == v25 )
            {
              v24 += 2;
              sub_CB6200(v40, (unsigned __int8 *)"\n", 1u);
              if ( s2 == v24 )
              {
LABEL_44:
                v17 = v66;
                break;
              }
            }
            else
            {
              *v25 = 10;
              ++*(_QWORD *)(v40 + 32);
LABEL_32:
              v24 += 2;
              if ( s2 == v24 )
                goto LABEL_44;
            }
          }
        }
        v35 = sub_CB72A0();
        v36 = (_BYTE *)v35[4];
        if ( (unsigned __int64)v36 >= v35[3] )
        {
          sub_CB5D20((__int64)v35, 10);
        }
        else
        {
          v35[4] = v36 + 1;
          *v36 = 10;
        }
        v37 = sub_CB72A0();
        v38 = (__m128i *)v37[4];
        if ( v37[3] - (_QWORD)v38 <= 0x7Du )
        {
          sub_CB6200(
            (__int64)v37,
            "Use -mcpu or -mtune to specify the target's processor.\n"
            "For example, clang --target=aarch64-unknown-linux-gnu -mcpu=cortex-a35\n",
            0x7Eu);
        }
        else
        {
          v39 = _mm_load_si128((const __m128i *)&xmmword_3F829A0);
          qmemcpy(&v38[7], "pu=cortex-a35\n", 14);
          *v38 = v39;
          v38[1] = _mm_load_si128((const __m128i *)&xmmword_3F829B0);
          v38[2] = _mm_load_si128((const __m128i *)&xmmword_3F829C0);
          v38[3] = _mm_load_si128((const __m128i *)&xmmword_3F829D0);
          v38[4] = _mm_load_si128((const __m128i *)&xmmword_3F829E0);
          v38[5] = _mm_load_si128((const __m128i *)&xmmword_3F829F0);
          v38[6] = _mm_load_si128((const __m128i *)&xmmword_3F82A00);
          v37[4] += 126LL;
        }
        byte_4F8A328 = 1;
      }
LABEL_18:
      v20 += 4;
      if ( v17 == v20 )
        goto LABEL_41;
    }
    v31 = *v20;
    v32 = (size_t)v20[1];
    v20 += 4;
    sub_E9FFE0((__int64)&v70, v31, v32, a13, a14);
  }
  while ( v17 != v20 );
LABEL_41:
  v17 = v68;
  v18 = v67;
LABEL_42:
  v33 = _mm_loadu_si128(&v71);
  v34 = v72;
  *(__m128i *)a1 = _mm_loadu_si128(&v70);
  *(_QWORD *)(a1 + 32) = v34;
  *(__m128i *)(a1 + 16) = v33;
LABEL_4:
  if ( v17 != v18 )
  {
    do
    {
      if ( *v18 != (char *)(v18 + 2) )
        j_j___libc_free_0(*v18, v18[2] + 1);
      v18 += 4;
    }
    while ( v17 != v18 );
    v18 = v67;
  }
  if ( v18 )
    j_j___libc_free_0(v18, v69 - (_QWORD)v18);
  return a1;
}
