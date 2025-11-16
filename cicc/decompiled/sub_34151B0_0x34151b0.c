// Function: sub_34151B0
// Address: 0x34151b0
//
void __fastcall sub_34151B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 i, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __m128i v14; // xmm1
  __int64 v15; // rax
  __m128i *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 *v21; // rcx
  unsigned int v22; // eax
  int v23; // r12d
  unsigned __int64 v24; // r8
  __int64 *v25; // r13
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // r14
  __int64 *v28; // rax
  int v29; // edi
  int v30; // r14d
  __int64 *j; // r12
  __int64 v32; // rdx
  __int64 v33; // r8
  unsigned __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // rsi
  int v37; // esi
  unsigned int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // r8
  __m128i *v41; // rdx
  __m128i si128; // xmm0
  __m128i *v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  unsigned __int64 v48; // [rsp+0h] [rbp-210h]
  unsigned __int64 v49; // [rsp+8h] [rbp-208h]
  __int64 v50; // [rsp+20h] [rbp-1F0h]
  int v51; // [rsp+40h] [rbp-1D0h]
  int v52; // [rsp+44h] [rbp-1CCh]
  __int64 v53; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 v54[2]; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v55; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v56; // [rsp+78h] [rbp-198h]
  __int64 v57; // [rsp+80h] [rbp-190h]
  __int64 v58; // [rsp+88h] [rbp-188h]
  __int64 v59[4]; // [rsp+90h] [rbp-180h] BYREF
  _QWORD *v60; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-158h]
  _QWORD v62[6]; // [rsp+C0h] [rbp-150h] BYREF
  __int64 *v63; // [rsp+F0h] [rbp-120h] BYREF
  __int64 v64; // [rsp+F8h] [rbp-118h]
  _BYTE v65[48]; // [rsp+100h] [rbp-110h] BYREF
  char *v66[2]; // [rsp+130h] [rbp-E0h] BYREF
  char v67; // [rsp+140h] [rbp-D0h] BYREF
  __int64 v68; // [rsp+148h] [rbp-C8h]
  __int64 v69; // [rsp+150h] [rbp-C0h]
  __int64 v70; // [rsp+158h] [rbp-B8h]
  __m128i v71; // [rsp+160h] [rbp-B0h] BYREF
  __int8 v72; // [rsp+170h] [rbp-A0h]
  __int64 v73; // [rsp+180h] [rbp-90h] BYREF
  void *s; // [rsp+188h] [rbp-88h]
  _BYTE v75[12]; // [rsp+190h] [rbp-80h]
  char v76; // [rsp+19Ch] [rbp-74h]
  char v77; // [rsp+1A0h] [rbp-70h] BYREF

  v6 = *(unsigned int *)(a1 + 752);
  v53 = a3;
  v7 = *(_QWORD *)(a1 + 736);
  if ( !(_DWORD)v6 )
    return;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = v7 + 80LL * v9;
  v11 = *(_QWORD *)v10;
  if ( a2 != *(_QWORD *)v10 )
  {
    for ( i = 1; ; i = (unsigned int)a6 )
    {
      if ( v11 == -4096 )
        return;
      a6 = (unsigned int)(i + 1);
      v9 = (v6 - 1) & (i + v9);
      v10 = v7 + 80LL * v9;
      v11 = *(_QWORD *)v10;
      if ( a2 == *(_QWORD *)v10 )
        break;
    }
  }
  if ( v10 == v7 + 80 * v6 )
    return;
  v12 = *(unsigned int *)(v10 + 16);
  v66[0] = &v67;
  v66[1] = (char *)0x100000000LL;
  if ( (_DWORD)v12 )
    sub_33C8070((__int64)v66, v10 + 8, v12, v7, i, a6);
  v13 = *(_QWORD *)(v10 + 48);
  v14 = _mm_loadu_si128((const __m128i *)(v10 + 56));
  v68 = *(_QWORD *)(v10 + 32);
  v15 = *(_QWORD *)(v10 + 40);
  v70 = v13;
  LOBYTE(v13) = *(_BYTE *)(v10 + 72);
  v69 = v15;
  v72 = v13;
  v71 = v14;
  if ( !v15 )
  {
    v16 = (__m128i *)sub_337D790(a1 + 728, &v53);
    sub_33C8150((__int64)v16, v66, v17, v18, v19, v20);
    v16[1].m128i_i64[1] = v68;
    v16[2].m128i_i64[0] = v69;
    v16[2].m128i_i64[1] = v70;
    v16[3] = _mm_loadu_si128(&v71);
    v16[4].m128i_i8[0] = v72;
    goto LABEL_8;
  }
  v55 = 0;
  v60 = v62;
  v61 = 0x600000001LL;
  v59[1] = (__int64)&v73;
  v21 = (__int64 *)v65;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v73 = 0;
  s = &v77;
  *(_QWORD *)v75 = 8;
  *(_DWORD *)&v75[8] = 0;
  v76 = 1;
  v59[2] = a1;
  v51 = 7;
  v52 = 16;
  v50 = a1 + 728;
  v54[0] = (__int64)&v60;
  v62[0] = a2;
  v54[1] = (__int64)&v55;
  v59[0] = (__int64)&v55;
  v22 = 1;
  v59[3] = (__int64)v66;
  v23 = 0;
  while ( 1 )
  {
    v24 = v22;
    v63 = (__int64 *)v65;
    v64 = 0x600000000LL;
    if ( v22 <= 6 )
    {
      v25 = (__int64 *)v65;
      v26 = 0;
      v27 = 0;
      v28 = (__int64 *)v65;
      goto LABEL_18;
    }
    sub_C8D5F0((__int64)&v63, v65, v22, 8u, v22, a6);
    v27 = (unsigned int)v64;
    if ( (unsigned int)v64 > (unsigned __int64)HIDWORD(v61) )
    {
      sub_C8D5F0((__int64)&v60, v62, (unsigned int)v64, 8u, v33, a6);
      v27 = (unsigned int)v64;
    }
    v24 = (unsigned int)v61;
    v26 = v27;
    v28 = v63;
    if ( (unsigned int)v61 <= v27 )
      v26 = (unsigned int)v61;
    if ( v26 )
    {
      a6 = 8 * v26;
      v34 = 0;
      while ( 1 )
      {
        v35 = &v28[v34 / 8];
        v36 = *v35;
        v21 = &v60[v34 / 8];
        v34 += 8LL;
        *v35 = *v21;
        *v21 = v36;
        if ( a6 == v34 )
          break;
        v28 = v63;
      }
      v27 = (unsigned int)v64;
      v24 = (unsigned int)v61;
      v28 = v63;
    }
    if ( v24 >= v27 )
    {
      v25 = &v28[v27];
LABEL_18:
      if ( v27 < v24 )
      {
        v29 = v27;
        a6 = (__int64)&v60[v26];
        if ( (_QWORD *)a6 != &v60[v24] )
        {
          v49 = v24;
          memcpy(v25, &v60[v26], 8 * v24 - 8 * v26);
          v29 = v64;
          v28 = v63;
          v24 = v49;
        }
        v24 -= v27;
        LODWORD(v61) = v26;
        LODWORD(v64) = v24 + v29;
        v25 = &v28[(unsigned int)(v24 + v29)];
      }
      goto LABEL_22;
    }
    a6 = 8 * v26;
    v37 = v24;
    v25 = &v28[v26];
    if ( &v28[v27] != v25 )
    {
      v48 = v24;
      memcpy(&v60[v24], v25, 8 * v27 - a6);
      v28 = v63;
      a6 = 8 * v26;
      v37 = v61;
      v24 = v48;
      v25 = &v63[v26];
    }
    LODWORD(v64) = v26;
    LODWORD(v61) = v37 + v27 - v24;
LABEL_22:
    v30 = v52 - v23;
    for ( j = v28; v25 != j; ++j )
    {
      v32 = *j;
      sub_3414EA0(v54, (__int64)v54, v32, v30, v24, a6);
    }
    if ( (unsigned __int8)sub_33F8F70(v59, (__int64)v59, v53, (__int64)v21, v24, a6) )
      break;
    if ( v63 != (__int64 *)v65 )
      _libc_free((unsigned __int64)v63);
    ++v73;
    if ( v76 )
      goto LABEL_51;
    v38 = 4 * (*(_DWORD *)&v75[4] - *(_DWORD *)&v75[8]);
    if ( v38 < 0x20 )
      v38 = 32;
    if ( v38 >= *(_DWORD *)v75 )
    {
      memset(s, -1, 8LL * *(unsigned int *)v75);
LABEL_51:
      *(_QWORD *)&v75[4] = 0;
      goto LABEL_52;
    }
    sub_C8C990((__int64)&v73, (__int64)v59);
LABEL_52:
    if ( !--v51 )
    {
      v39 = sub_CB72A0();
      v40 = v50;
      v41 = (__m128i *)v39[4];
      if ( v39[3] - (_QWORD)v41 <= 0x3Eu )
      {
        sub_CB6200((__int64)v39, "warning: incomplete propagation of SelectionDAG::NodeExtraInfo\n", 0x3Fu);
        v40 = v50;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_44DF840);
        qmemcpy(&v41[3], ":NodeExtraInfo\n", 15);
        *v41 = si128;
        v41[1] = _mm_load_si128((const __m128i *)&xmmword_44DF850);
        v41[2] = _mm_load_si128((const __m128i *)&xmmword_44DF860);
        v39[4] += 63LL;
      }
      v43 = (__m128i *)sub_337D790(v40, &v53);
      sub_33C8150((__int64)v43, v66, v44, v45, v46, v47);
      v43[1].m128i_i64[1] = v68;
      v43[2].m128i_i64[0] = v69;
      v43[2].m128i_i64[1] = v70;
      v43[3] = _mm_loadu_si128(&v71);
      v43[4].m128i_i8[0] = v72;
      goto LABEL_27;
    }
    v23 = v52;
    v22 = v61;
    v52 *= 2;
  }
  if ( v63 != (__int64 *)v65 )
    _libc_free((unsigned __int64)v63);
LABEL_27:
  if ( !v76 )
    _libc_free((unsigned __int64)s);
  sub_C7D6A0(v56, 8LL * (unsigned int)v58, 8);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
LABEL_8:
  if ( v66[0] != &v67 )
    _libc_free((unsigned __int64)v66[0]);
}
