// Function: sub_2D0B540
// Address: 0x2d0b540
//
__int64 __fastcall sub_2D0B540(__int64 a1, char a2)
{
  __int64 v3; // rsi
  char *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r12d
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r13
  _QWORD *v14; // rax
  _DWORD *v15; // rdx
  __int64 v16; // r8
  const char *v17; // rax
  size_t v18; // rdx
  __int64 v19; // r8
  unsigned __int8 *v20; // rsi
  _BYTE *v21; // rax
  _BYTE *v22; // rdi
  void *v23; // rax
  __int64 v24; // r9
  __m128i *v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r12
  _QWORD *v28; // rax
  __m128i *v29; // rdx
  __int64 v30; // r8
  __m128i si128; // xmm0
  __int64 v32; // rdx
  __int64 v33; // rsi
  unsigned int v34; // eax
  _QWORD *v35; // rax
  __int64 v36; // r9
  __int64 v37; // r8
  size_t v38; // rdx
  const void *v39; // rsi
  int v40; // eax
  size_t v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rsi
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  unsigned __int64 v46; // r13
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  __int64 v49; // [rsp+0h] [rbp-E0h]
  unsigned __int8 v50; // [rsp+10h] [rbp-D0h]
  __int64 v51; // [rsp+10h] [rbp-D0h]
  __int64 v53; // [rsp+20h] [rbp-C0h]
  __int64 v54; // [rsp+20h] [rbp-C0h]
  __int64 v55; // [rsp+20h] [rbp-C0h]
  __int64 v56; // [rsp+20h] [rbp-C0h]
  size_t v57; // [rsp+20h] [rbp-C0h]
  __int64 i; // [rsp+28h] [rbp-B8h]
  __int64 v59; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v60; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v61; // [rsp+48h] [rbp-98h]
  __int64 v62; // [rsp+50h] [rbp-90h]
  unsigned int v63; // [rsp+58h] [rbp-88h]
  __int64 v64[2]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v65[112]; // [rsp+70h] [rbp-70h] BYREF

  v3 = *(_QWORD *)a1;
  v64[0] = (__int64)v65;
  v64[1] = 0x800000000LL;
  sub_2D06F50((__int64)v64, v3);
  v8 = sub_2D06650(v64, *(_QWORD *)(a1 + 8), v4, v5, v6, v7);
  if ( (_BYTE)v8 )
    goto LABEL_2;
  v10 = *(_QWORD *)(a1 + 120);
  ++*(_QWORD *)(a1 + 112);
  v11 = *(_DWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 120) = 0;
  v61 = (_QWORD *)v10;
  v12 = *(_QWORD *)(a1 + 128);
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  v63 = v11;
  v60 = 1;
  v62 = v12;
  sub_CE6110((_QWORD *)a1);
  v50 = 1;
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  for ( i = *(_QWORD *)a1 + 72LL; i != v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    v32 = *(_QWORD *)(a1 + 16);
    if ( v13 )
    {
      v59 = v13 - 24;
      v33 = (unsigned int)(*(_DWORD *)(v13 + 20) + 1);
      v34 = *(_DWORD *)(v13 + 20) + 1;
    }
    else
    {
      v59 = 0;
      v33 = 0;
      v34 = 0;
    }
    if ( v34 >= *(_DWORD *)(v32 + 32) || !*(_QWORD *)(*(_QWORD *)(v32 + 24) + 8 * v33) )
      continue;
    v55 = *sub_CE3FC0(a1 + 112, &v59);
    v35 = sub_CE3FC0((__int64)&v60, &v59);
    v36 = v55;
    v37 = *v35;
    if ( *(_DWORD *)(*v35 + 8LL) == *(_DWORD *)(v55 + 8)
      && *(_DWORD *)(v37 + 12) == *(_DWORD *)(v55 + 12)
      && *(_DWORD *)(v37 + 88) == *(_DWORD *)(v55 + 88) )
    {
      v38 = 8LL * *(unsigned int *)(v37 + 32);
      if ( !v38
        || (v39 = *(const void **)(v55 + 24),
            v49 = v55,
            v56 = *v35,
            v40 = memcmp(*(const void **)(v37 + 24), v39, v38),
            v37 = v56,
            v36 = v49,
            !v40) )
      {
        if ( *(_DWORD *)(v37 + 160) == *(_DWORD *)(v36 + 160) )
        {
          v41 = 8LL * *(unsigned int *)(v37 + 104);
          if ( !v41 || !memcmp(*(const void **)(v37 + 96), *(const void **)(v36 + 96), v41) )
            continue;
        }
      }
    }
    if ( !a2 )
    {
      v50 = 0;
      break;
    }
    v14 = sub_CB72A0();
    v15 = (_DWORD *)v14[4];
    v16 = (__int64)v14;
    if ( v14[3] - (_QWORD)v15 <= 3u )
    {
      v16 = sub_CB6200((__int64)v14, (unsigned __int8 *)"BB: ", 4u);
    }
    else
    {
      *v15 = 540688962;
      v14[4] += 4LL;
    }
    v53 = v16;
    v17 = sub_BD5D20(v59);
    v19 = v53;
    v20 = (unsigned __int8 *)v17;
    v21 = *(_BYTE **)(v53 + 24);
    v22 = *(_BYTE **)(v53 + 32);
    if ( v21 - v22 < v18 )
    {
      v19 = sub_CB6200(v53, v20, v18);
      v22 = *(_BYTE **)(v19 + 32);
      if ( v22 == *(_BYTE **)(v19 + 24) )
        goto LABEL_37;
    }
    else
    {
      if ( v18 )
      {
        v51 = v53;
        v57 = v18;
        memcpy(v22, v20, v18);
        v19 = v51;
        v22 = (_BYTE *)(*(_QWORD *)(v51 + 32) + v57);
        v21 = *(_BYTE **)(v51 + 24);
        *(_QWORD *)(v51 + 32) = v22;
      }
      if ( v22 == v21 )
      {
LABEL_37:
        sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
        goto LABEL_15;
      }
    }
    *v22 = 10;
    ++*(_QWORD *)(v19 + 32);
LABEL_15:
    v54 = *sub_CE3FC0(a1 + 112, &v59);
    v23 = sub_CB72A0();
    v24 = v54;
    v25 = (__m128i *)*((_QWORD *)v23 + 4);
    v26 = (__int64)v23;
    if ( *((_QWORD *)v23 + 3) - (_QWORD)v25 <= 0xFu )
    {
      v42 = sub_CB6200((__int64)v23, "Correct RP Info\n", 0x10u);
      v24 = v54;
      v26 = v42;
    }
    else
    {
      *v25 = _mm_load_si128((const __m128i *)&xmmword_42EAB10);
      *((_QWORD *)v23 + 4) += 16LL;
    }
    sub_2D06940(a1, v26, v24);
    v27 = *sub_CE3FC0((__int64)&v60, &v59);
    v28 = sub_CB72A0();
    v29 = (__m128i *)v28[4];
    v30 = (__int64)v28;
    if ( v28[3] - (_QWORD)v29 <= 0x11u )
    {
      v30 = sub_CB6200((__int64)v28, "Incorrect RP Info\n", 0x12u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EAB20);
      v29[1].m128i_i16[0] = 2671;
      *v29 = si128;
      v28[4] += 18LL;
    }
    sub_2D06940(a1, v30, v27);
    v50 = 0;
  }
  v43 = v63;
  if ( v63 )
  {
    v44 = v61;
    v45 = &v61[2 * v63];
    do
    {
      if ( *v44 != -8192 && *v44 != -4096 )
      {
        v46 = v44[1];
        if ( v46 )
        {
          v47 = *(_QWORD *)(v46 + 96);
          if ( v47 != v46 + 112 )
            _libc_free(v47);
          v48 = *(_QWORD *)(v46 + 24);
          if ( v48 != v46 + 40 )
            _libc_free(v48);
          j_j___libc_free_0(v46);
        }
      }
      v44 += 2;
    }
    while ( v45 != v44 );
    v43 = v63;
  }
  sub_C7D6A0((__int64)v61, 16 * v43, 8);
  v8 = v50;
LABEL_2:
  if ( (_BYTE *)v64[0] != v65 )
    _libc_free(v64[0]);
  return v8;
}
