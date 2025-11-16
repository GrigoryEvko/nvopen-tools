// Function: sub_3030EB0
// Address: 0x3030eb0
//
void __fastcall sub_3030EB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rdx
  const __m128i *v12; // rbx
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  int v15; // eax
  __m128i v16; // xmm1
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __int64 v19; // r15
  unsigned __int64 v20; // rcx
  __int64 v21; // r13
  __int64 i; // rax
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  unsigned int *v28; // rdx
  __int64 v29; // rdi
  unsigned __int64 v30; // r12
  __int64 v31; // rdx
  __int16 v32; // si
  __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  char v38; // al
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 *v41; // r10
  unsigned __int64 v42; // rdx
  _QWORD *v43; // rax
  unsigned __int64 v44; // rsi
  int v45; // eax
  int v46; // edx
  int v47; // r9d
  __int64 v48; // rax
  __int64 v49; // r9
  __int64 v50; // r10
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // r8
  int v55; // r12d
  int v56; // r15d
  unsigned __int64 v57; // r13
  unsigned __int64 v58; // rbx
  __int64 *v59; // rax
  __int128 v60; // [rsp-10h] [rbp-1B0h]
  unsigned __int64 v61; // [rsp+8h] [rbp-198h]
  __m128i v62; // [rsp+10h] [rbp-190h] BYREF
  __int64 v63; // [rsp+20h] [rbp-180h]
  _OWORD *v64; // [rsp+28h] [rbp-178h]
  int v65; // [rsp+34h] [rbp-16Ch]
  _BYTE *v66; // [rsp+38h] [rbp-168h]
  __int64 *v67; // [rsp+40h] [rbp-160h]
  __int64 m128i_i64; // [rsp+48h] [rbp-158h]
  __int64 v69; // [rsp+50h] [rbp-150h]
  __int64 *v70; // [rsp+58h] [rbp-148h]
  __int64 v71; // [rsp+60h] [rbp-140h] BYREF
  int v72; // [rsp+68h] [rbp-138h]
  __int64 v73; // [rsp+70h] [rbp-130h] BYREF
  __int64 v74; // [rsp+78h] [rbp-128h]
  __int64 v75; // [rsp+80h] [rbp-120h] BYREF
  char v76; // [rsp+88h] [rbp-118h]
  __int64 v77; // [rsp+90h] [rbp-110h]
  __int64 v78; // [rsp+98h] [rbp-108h]
  _BYTE *v79; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-F8h]
  _BYTE v81[48]; // [rsp+B0h] [rbp-F0h] BYREF
  _OWORD *v82; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-B8h]
  _OWORD v84[11]; // [rsp+F0h] [rbp-B0h] BYREF

  v9 = *(_QWORD *)(a1 + 80);
  v69 = a1;
  v71 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v71, v9, 1);
  v82 = v84;
  v10 = *(_DWORD *)(v69 + 72);
  v11 = *(_QWORD *)(v69 + 40);
  v64 = v84;
  v83 = 0x800000000LL;
  v12 = *(const __m128i **)(v69 + 48);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = _mm_loadu_si128((const __m128i *)(v11 + 80));
  v72 = v10;
  v15 = *(_DWORD *)(v69 + 68);
  v16 = _mm_loadu_si128((const __m128i *)v11);
  v66 = v81;
  v17 = _mm_loadu_si128((const __m128i *)(v11 + 120));
  v18 = _mm_loadu_si128((const __m128i *)(v11 + 160));
  LODWORD(v83) = 5;
  v79 = v81;
  v65 = v15 - 1;
  v80 = 0x300000000LL;
  v84[0] = v16;
  v84[1] = v13;
  v84[2] = v14;
  v84[3] = v17;
  v84[4] = v18;
  if ( v15 - 1 <= 0 )
  {
    v40 = 0;
    v41 = &v71;
  }
  else
  {
    v63 = a3;
    v19 = v6;
    v67 = &v75;
    v20 = 3;
    v21 = 200;
    v70 = a2;
    m128i_i64 = (__int64)v12[(unsigned int)(v15 - 2) + 1].m128i_i64;
    for ( i = 0; ; i = (unsigned int)v80 )
    {
      v28 = (unsigned int *)(v21 + v11);
      v29 = *(_QWORD *)(*(_QWORD *)v28 + 96LL);
      v30 = *(_QWORD *)(v29 + 24);
      if ( *(_DWORD *)(v29 + 32) > 0x40u )
        v30 = *(_QWORD *)v30;
      v31 = *(_QWORD *)(*(_QWORD *)v28 + 48LL) + 16LL * v28[2];
      v32 = *(_WORD *)v31;
      v33 = *(_QWORD *)(v31 + 8);
      LOWORD(v73) = v32;
      v74 = v33;
      v34 = i + 1;
      if ( v12->m128i_i16[0] == 5 )
      {
        LOWORD(v19) = 6;
        if ( v34 > v20 )
        {
          sub_C8D5F0((__int64)&v79, v66, v34, 0x10u, a5, a6);
          i = (unsigned int)v80;
        }
        v35 = &v79[16 * i];
        *v35 = v19;
        v35[1] = 0;
        LODWORD(v80) = v80 + 1;
        if ( (_WORD)v73 )
        {
          if ( (_WORD)v73 == 1 || (unsigned __int16)(v73 - 504) <= 7u )
            BUG();
          v39 = 16LL * ((unsigned __int16)v73 - 1);
          v37 = *(_QWORD *)&byte_444C4A0[v39];
          v38 = byte_444C4A0[v39 + 8];
        }
        else
        {
          v77 = sub_3007260((__int64)&v73);
          v78 = v36;
          v37 = v77;
          v38 = v78;
        }
        v75 = v37;
        v76 = v38;
        v30 |= 1LL << ((unsigned __int8)sub_CA1930(v67) - 1);
      }
      else
      {
        v23 = _mm_loadu_si128(v12);
        if ( v34 > v20 )
        {
          v62 = v23;
          sub_C8D5F0((__int64)&v79, v66, v34, 0x10u, a5, a6);
          i = (unsigned int)v80;
          v23 = _mm_load_si128(&v62);
        }
        *(__m128i *)&v79[16 * i] = v23;
        LODWORD(v80) = v80 + 1;
      }
      a5 = sub_3400BD0((_DWORD)v70, v30, (unsigned int)&v71, v73, v74, 1, 0);
      v24 = (unsigned int)v83;
      a6 = v25;
      v26 = (unsigned int)v83 + 1LL;
      if ( v26 > HIDWORD(v83) )
      {
        v62.m128i_i64[0] = a5;
        v62.m128i_i64[1] = a6;
        sub_C8D5F0((__int64)&v82, v64, v26, 0x10u, a5, a6);
        v24 = (unsigned int)v83;
        a6 = v62.m128i_i64[1];
        a5 = v62.m128i_i64[0];
      }
      v27 = (__int64 *)&v82[v24];
      v21 += 40;
      ++v12;
      *v27 = a5;
      v27[1] = a6;
      LODWORD(v83) = v83 + 1;
      if ( v12 == (const __m128i *)m128i_i64 )
        break;
      v20 = HIDWORD(v80);
      v11 = *(_QWORD *)(v69 + 40);
    }
    v40 = (unsigned int)v80;
    v41 = &v71;
    a3 = v63;
    a2 = v70;
    v42 = (unsigned int)v80 + 1LL;
    if ( v42 > HIDWORD(v80) )
    {
      v70 = &v71;
      sub_C8D5F0((__int64)&v79, v66, v42, 0x10u, a5, a6);
      v40 = (unsigned int)v80;
      v41 = v70;
    }
  }
  v43 = &v79[16 * v40];
  v70 = v41;
  *v43 = 1;
  v44 = (unsigned __int64)v79;
  v43[1] = 0;
  LODWORD(v80) = v80 + 1;
  v45 = sub_33E5830(a2, v44);
  *((_QWORD *)&v60 + 1) = (unsigned int)v83;
  *(_QWORD *)&v60 = v82;
  v48 = sub_3411630((_DWORD)a2, 317, (_DWORD)v70, v45, v46, v47, v60);
  v50 = (__int64)v70;
  v51 = v48;
  if ( v65 >= 0 )
  {
    v52 = *(unsigned int *)(a3 + 8);
    v53 = a3;
    v54 = v51;
    v55 = 0;
    v56 = v65;
    v57 = v61;
    do
    {
      v58 = v57 & 0xFFFFFFFF00000000LL | (unsigned int)v55;
      v57 = v58;
      if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(v53 + 12) )
      {
        m128i_i64 = v50;
        v69 = v54;
        v70 = (__int64 *)v53;
        sub_C8D5F0(v53, (const void *)(v53 + 16), v52 + 1, 0x10u, v54, v49);
        v53 = (__int64)v70;
        v50 = m128i_i64;
        v54 = v69;
        v52 = *((unsigned int *)v70 + 2);
      }
      v59 = (__int64 *)(*(_QWORD *)v53 + 16 * v52);
      ++v55;
      *v59 = v54;
      v59[1] = v58;
      v52 = (unsigned int)(*(_DWORD *)(v53 + 8) + 1);
      *(_DWORD *)(v53 + 8) = v52;
    }
    while ( v56 >= v55 );
  }
  if ( v79 != v66 )
  {
    v70 = (__int64 *)v50;
    _libc_free((unsigned __int64)v79);
    v50 = (__int64)v70;
  }
  if ( v82 != v64 )
  {
    v70 = (__int64 *)v50;
    _libc_free((unsigned __int64)v82);
    v50 = (__int64)v70;
  }
  if ( v71 )
    sub_B91220(v50, v71);
}
