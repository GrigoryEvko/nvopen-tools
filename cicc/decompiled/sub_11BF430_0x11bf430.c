// Function: sub_11BF430
// Address: 0x11bf430
//
__int64 __fastcall sub_11BF430(__int64 ***a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 **v3; // r12
  __m128i *v4; // rsi
  char *v5; // rcx
  __int64 v6; // rax
  __m128i *v7; // r15
  unsigned __int64 v8; // r9
  __m128i *v9; // rbx
  int v10; // edx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rbx
  __m128i *v18; // r15
  char *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  size_t v22; // rbx
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // r13
  __m128i *v30; // rbx
  __int64 v31; // r12
  __int64 v32; // r15
  __int8 *v33; // r14
  __int8 *v34; // rcx
  int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned __int64 v38; // rsi
  _QWORD *v39; // rax
  __m128i *v40; // rbx
  __m128i *v41; // r13
  __int64 v42; // rdi
  int v44; // ebx
  char *v45; // rbx
  __int64 *v46; // [rsp+0h] [rbp-2E0h]
  __int64 v47; // [rsp+8h] [rbp-2D8h]
  char *v48; // [rsp+20h] [rbp-2C0h]
  int v49; // [rsp+20h] [rbp-2C0h]
  __int64 **v50; // [rsp+30h] [rbp-2B0h]
  int v51; // [rsp+30h] [rbp-2B0h]
  _QWORD *v52; // [rsp+38h] [rbp-2A8h]
  unsigned __int64 v53; // [rsp+58h] [rbp-288h] BYREF
  __m128i *v54; // [rsp+60h] [rbp-280h] BYREF
  __int64 v55; // [rsp+68h] [rbp-278h]
  _QWORD src[2]; // [rsp+70h] [rbp-270h] BYREF
  __m128i *v57; // [rsp+80h] [rbp-260h] BYREF
  __int64 v58; // [rsp+88h] [rbp-258h]
  __m128i v59; // [rsp+90h] [rbp-250h] BYREF
  _QWORD v60[2]; // [rsp+A0h] [rbp-240h] BYREF
  __m128i v61; // [rsp+B0h] [rbp-230h] BYREF
  char *v62; // [rsp+C0h] [rbp-220h]
  char *v63; // [rsp+C8h] [rbp-218h]
  char *v64; // [rsp+D0h] [rbp-210h]
  __m128i *v65; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v66; // [rsp+E8h] [rbp-1F8h]
  _BYTE v67[496]; // [rsp+F0h] [rbp-1F0h] BYREF

  v1 = 0;
  if ( !*((_DWORD *)a1 + 56) )
    return v1;
  v2 = sub_B6E160((__int64 *)*a1, 0xBu, 0, 0);
  v3 = a1[27];
  v47 = v2;
  v46 = **a1;
  v65 = (__m128i *)v67;
  v66 = 0x800000000LL;
  v50 = &v3[3 * *((unsigned int *)a1 + 56)];
  if ( v3 != v50 )
  {
    v52 = v60;
    do
    {
      v54 = (__m128i *)src;
      v55 = 0x200000000LL;
      if ( *v3 )
      {
        src[0] = *v3;
        LODWORD(v55) = 1;
      }
      v16 = (__int64)v3[2];
      if ( v16 )
      {
        v23 = sub_BCB2E0(**a1);
        v24 = sub_ACD640(v23, v16, 0);
        v27 = (unsigned int)v55;
        v28 = (unsigned int)v55 + 1LL;
        if ( v28 > HIDWORD(v55) )
        {
          sub_C8D5F0((__int64)&v54, src, v28, 8u, v25, v26);
          v27 = (unsigned int)v55;
        }
        v54->m128i_i64[v27] = v24;
        v18 = v54;
        v17 = (unsigned int)(v55 + 1);
        LODWORD(v55) = v55 + 1;
      }
      else
      {
        v17 = (unsigned int)v55;
        v18 = (__m128i *)src;
      }
      v19 = sub_A6FBB0(*((_DWORD *)v3 + 2));
      v57 = &v59;
      v4 = (__m128i *)v19;
      sub_11BE0C0((__int64 *)&v57, v19, (__int64)&v19[v20]);
      v60[0] = &v61;
      if ( v57 == &v59 )
      {
        v61 = _mm_load_si128(&v59);
      }
      else
      {
        v60[0] = v57;
        v61.m128i_i64[0] = v59.m128i_i64[0];
      }
      v21 = v58;
      v22 = 8 * v17;
      v57 = &v59;
      v58 = 0;
      v60[1] = v21;
      v59.m128i_i8[0] = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      if ( v22 )
      {
        v4 = v18;
        v62 = (char *)sub_22077B0(v22);
        v64 = &v62[v22];
        v48 = &v62[v22];
        memcpy(v62, v18, v22);
        v5 = v48;
      }
      else
      {
        v5 = 0;
      }
      v6 = (unsigned int)v66;
      v63 = v5;
      v7 = v65;
      v8 = (unsigned int)v66 + 1LL;
      v9 = (__m128i *)v60;
      v10 = v66;
      if ( v8 > HIDWORD(v66) )
      {
        if ( v65 > (__m128i *)v60 || v60 >= &v65->m128i_i64[7 * (unsigned int)v66] )
        {
          v4 = (__m128i *)sub_C8D7D0((__int64)&v65, (__int64)v67, (unsigned int)v66 + 1LL, 0x38u, &v53, v8);
          v7 = v4;
          sub_B56820((__int64)&v65, v4);
          v44 = v53;
          if ( v65 != (__m128i *)v67 )
            _libc_free(v65, v4);
          v6 = (unsigned int)v66;
          HIDWORD(v66) = v44;
          v65 = v4;
          v9 = (__m128i *)v60;
          v10 = v66;
        }
        else
        {
          v45 = (char *)((char *)v60 - (char *)v65);
          v4 = (__m128i *)sub_C8D7D0((__int64)&v65, (__int64)v67, (unsigned int)v66 + 1LL, 0x38u, &v53, v8);
          v7 = v4;
          sub_B56820((__int64)&v65, v4);
          if ( v65 == (__m128i *)v67 )
          {
            v65 = v4;
            HIDWORD(v66) = v53;
          }
          else
          {
            v49 = v53;
            _libc_free(v65, v4);
            v65 = v4;
            HIDWORD(v66) = v49;
          }
          v6 = (unsigned int)v66;
          v9 = (__m128i *)&v45[(_QWORD)v4];
          v10 = v66;
        }
      }
      v11 = (__m128i *)((char *)v7 + 56 * v6);
      if ( v11 )
      {
        v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
        if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
        {
          v11[1] = _mm_loadu_si128(v9 + 1);
        }
        else
        {
          v11->m128i_i64[0] = v9->m128i_i64[0];
          v11[1].m128i_i64[0] = v9[1].m128i_i64[0];
        }
        v12 = v9->m128i_i64[1];
        v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
        v9->m128i_i64[1] = 0;
        v11->m128i_i64[1] = v12;
        v13 = v9[2].m128i_i64[0];
        v9[1].m128i_i8[0] = 0;
        v11[2].m128i_i64[0] = v13;
        v14 = v9[2].m128i_i64[1];
        v9[2].m128i_i64[0] = 0;
        v11[2].m128i_i64[1] = v14;
        v15 = v9[3].m128i_i64[0];
        v9[2].m128i_i64[1] = 0;
        v9[3].m128i_i64[0] = 0;
        v11[3].m128i_i64[0] = v15;
        v10 = v66;
      }
      LODWORD(v66) = v10 + 1;
      if ( v62 )
      {
        v4 = (__m128i *)(v64 - v62);
        j_j___libc_free_0(v62, v64 - v62);
      }
      if ( (__m128i *)v60[0] != &v61 )
      {
        v4 = (__m128i *)(v61.m128i_i64[0] + 1);
        j_j___libc_free_0(v60[0], v61.m128i_i64[0] + 1);
      }
      if ( v57 != &v59 )
      {
        v4 = (__m128i *)(v59.m128i_i64[0] + 1);
        j_j___libc_free_0(v57, v59.m128i_i64[0] + 1);
      }
      if ( v54 != (__m128i *)src )
        _libc_free(v54, v4);
      v3 += 3;
    }
    while ( v50 != v3 );
    v29 = (unsigned int)v66;
    v30 = v65;
    LOWORD(v62) = 257;
    v31 = (unsigned int)(16 * v66);
    LOBYTE(v52) = 16 * (_DWORD)v66 != 0;
    v32 = 0;
    v33 = &v65->m128i_i8[56 * (unsigned int)v66];
    v57 = (__m128i *)sub_ACD6D0(v46);
    if ( !v47 )
      goto LABEL_33;
    goto LABEL_32;
  }
  LOWORD(v62) = 257;
  v57 = (__m128i *)sub_ACD6D0(v46);
  if ( v47 )
  {
    v30 = (__m128i *)v67;
    LOBYTE(v52) = 0;
    v31 = 0;
    v29 = 0;
    v33 = v67;
LABEL_32:
    v32 = *(_QWORD *)(v47 + 24);
LABEL_33:
    if ( v33 == (__int8 *)v30 )
    {
      v51 = 2;
      v37 = 2;
    }
    else
    {
      v34 = (__int8 *)v30;
      v35 = 0;
      do
      {
        v36 = *((_QWORD *)v34 + 5) - *((_QWORD *)v34 + 4);
        v34 += 56;
        v35 += v36 >> 3;
      }
      while ( v33 != v34 );
      v37 = (unsigned int)(v35 + 2);
      v51 = v37 & 0x7FFFFFF;
    }
    goto LABEL_37;
  }
  LOBYTE(v52) = 0;
  v32 = 0;
  v31 = 0;
  v29 = 0;
  v51 = 2;
  v30 = (__m128i *)v67;
  v37 = 2;
LABEL_37:
  v38 = (v31 << 32) | v37;
  v39 = sub_BD2CC0(88, v38);
  v1 = (__int64)v39;
  if ( v39 )
  {
    sub_B44260((__int64)v39, **(_QWORD **)(v32 + 16), 56, v51 | ((_DWORD)v52 << 28), 0, 0);
    v38 = v32;
    *(_QWORD *)(v1 + 72) = 0;
    sub_B4A290(v1, v32, v47, (__int64 *)&v57, 1, (__int64)v60, (__int64)v30, v29);
  }
  v40 = v65;
  v41 = (__m128i *)((char *)v65 + 56 * (unsigned int)v66);
  if ( v65 != v41 )
  {
    do
    {
      v42 = v41[-2].m128i_i64[1];
      v41 = (__m128i *)((char *)v41 - 56);
      if ( v42 )
      {
        v38 = v41[3].m128i_i64[0] - v42;
        j_j___libc_free_0(v42, v38);
      }
      if ( (__m128i *)v41->m128i_i64[0] != &v41[1] )
      {
        v38 = v41[1].m128i_i64[0] + 1;
        j_j___libc_free_0(v41->m128i_i64[0], v38);
      }
    }
    while ( v40 != v41 );
    v41 = v65;
  }
  if ( v41 != (__m128i *)v67 )
    _libc_free(v41, v38);
  return v1;
}
