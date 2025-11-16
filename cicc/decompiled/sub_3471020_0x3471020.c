// Function: sub_3471020
// Address: 0x3471020
//
unsigned __int8 *__fastcall sub_3471020(__int64 a1, __int64 a2, _QWORD *a3, __m128i a4)
{
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  unsigned __int8 *result; // rax
  unsigned __int16 v10; // ax
  __int64 v11; // rsi
  int v12; // eax
  unsigned int v13; // esi
  char v14; // al
  __int64 v15; // r10
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int128 *v19; // rax
  __int128 *v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __m128i v24; // xmm0
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned int v28; // ebx
  unsigned __int8 *v29; // r13
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // r9
  __int128 v33; // rax
  __int64 v34; // r9
  __int128 v35; // [rsp-20h] [rbp-1A0h]
  __int128 v36; // [rsp-20h] [rbp-1A0h]
  __int128 v37; // [rsp-10h] [rbp-190h]
  __m128i v38; // [rsp+0h] [rbp-180h] BYREF
  _BYTE **v39; // [rsp+10h] [rbp-170h]
  _BYTE *v40; // [rsp+18h] [rbp-168h]
  _BYTE *v41; // [rsp+20h] [rbp-160h]
  unsigned int v42; // [rsp+2Ch] [rbp-154h]
  __m128i *v43; // [rsp+30h] [rbp-150h]
  __int64 v44; // [rsp+38h] [rbp-148h]
  __int64 v45; // [rsp+50h] [rbp-130h] BYREF
  __int64 v46; // [rsp+58h] [rbp-128h]
  __int64 v47; // [rsp+60h] [rbp-120h] BYREF
  int v48; // [rsp+68h] [rbp-118h]
  unsigned int v49; // [rsp+70h] [rbp-110h] BYREF
  __int64 v50; // [rsp+78h] [rbp-108h]
  unsigned int v51; // [rsp+80h] [rbp-100h] BYREF
  __int64 v52; // [rsp+88h] [rbp-F8h]
  __m128i v53; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v54; // [rsp+A0h] [rbp-E0h] BYREF
  _BYTE *v55; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-C8h]
  _BYTE v57[64]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE *v58; // [rsp+100h] [rbp-80h] BYREF
  __int64 v59; // [rsp+108h] [rbp-78h]
  _BYTE v60[112]; // [rsp+110h] [rbp-70h] BYREF

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v45) = v7;
  v46 = v8;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
      return 0;
    v10 = word_4456340[v7 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v45) )
      return 0;
    LOBYTE(v10) = sub_3007240((__int64)&v45);
  }
  if ( (v10 & 1) != 0 )
    return 0;
  sub_33D0340((__int64)&v49, (__int64)a3, &v45);
  if ( (_WORD)v51 != (_WORD)v49 || !(_WORD)v49 || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v49 + 112) )
    return 0;
  v11 = *(_QWORD *)(a2 + 80);
  v47 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v47, v11, 1);
  v12 = *(_DWORD *)(a2 + 72);
  v13 = *(_DWORD *)(a2 + 24);
  LODWORD(v43) = v49;
  v44 = v50;
  v48 = v12;
  v42 = v13;
  v14 = sub_328C7F0(a1, v13, v49, v50, 0);
  v15 = v44;
  v16 = (unsigned int)v43;
  if ( v14 )
  {
    v18 = *(unsigned int *)(a2 + 64);
    v41 = v57;
    v55 = v57;
    v56 = 0x400000000LL;
    v59 = 0x400000000LL;
    v19 = *(__int128 **)(a2 + 40);
    v40 = v60;
    v58 = v60;
    v44 = (__int64)v19 + 40 * v18;
    if ( v19 != (__int128 *)v44 )
    {
      v20 = v19;
      v43 = &v53;
      v39 = &v55;
      do
      {
        sub_3408290((__int64)v43, a3, v20, (__int64)&v47, &v49, &v51, a4);
        v23 = (unsigned int)v56;
        v24 = _mm_load_si128(&v53);
        v25 = (unsigned int)v56 + 1LL;
        if ( v25 > HIDWORD(v56) )
        {
          v38 = v24;
          sub_C8D5F0((__int64)v39, v41, v25, 0x10u, v21, v22);
          v23 = (unsigned int)v56;
          v24 = _mm_load_si128(&v38);
        }
        *(__m128i *)&v55[16 * v23] = v24;
        v26 = (unsigned int)v59;
        LODWORD(v56) = v56 + 1;
        a4 = _mm_load_si128(&v54);
        v27 = (unsigned int)v59 + 1LL;
        if ( v27 > HIDWORD(v59) )
        {
          v38 = a4;
          sub_C8D5F0((__int64)&v58, v40, v27, 0x10u, v21, v22);
          v26 = (unsigned int)v59;
          a4 = _mm_load_si128(&v38);
        }
        v20 = (__int128 *)((char *)v20 + 40);
        *(__m128i *)&v58[16 * v26] = a4;
        LODWORD(v59) = v59 + 1;
      }
      while ( (__int128 *)v44 != v20 );
      v16 = v49;
      v15 = v50;
    }
    v28 = v42;
    *((_QWORD *)&v37 + 1) = (unsigned int)v56;
    *(_QWORD *)&v37 = v55;
    v29 = sub_33FC220(a3, v42, (__int64)&v47, (unsigned int)v16, v15, v16, v37);
    v31 = v30;
    *((_QWORD *)&v35 + 1) = (unsigned int)v59;
    *(_QWORD *)&v35 = v58;
    *(_QWORD *)&v33 = sub_33FC220(a3, v28, (__int64)&v47, v51, v52, v32, v35);
    *((_QWORD *)&v36 + 1) = v31;
    *(_QWORD *)&v36 = v29;
    result = sub_3406EB0(a3, 0x9Fu, (__int64)&v47, (unsigned int)v45, v46, v34, v36, v33);
    if ( v58 != v40 )
    {
      v43 = (__m128i *)v17;
      v44 = (__int64)result;
      _libc_free((unsigned __int64)v58);
      v17 = (__int64)v43;
      result = (unsigned __int8 *)v44;
    }
    if ( v55 != v41 )
    {
      v43 = (__m128i *)v17;
      v44 = (__int64)result;
      _libc_free((unsigned __int64)v55);
      v17 = (__int64)v43;
      result = (unsigned __int8 *)v44;
    }
  }
  else
  {
    result = 0;
    v17 = 0;
  }
  if ( v47 )
  {
    v43 = (__m128i *)v17;
    v44 = (__int64)result;
    sub_B91220((__int64)&v47, v47);
    return (unsigned __int8 *)v44;
  }
  return result;
}
