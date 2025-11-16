// Function: sub_33E9C90
// Address: 0x33e9c90
//
__m128i *__fastcall sub_33E9C90(
        __int64 *a1,
        int a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        const __m128i *a13)
{
  unsigned __int16 v15; // r15
  __int64 v16; // r14
  unsigned __int16 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r10
  __int64 v20; // rdx
  __int64 v21; // r9
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned __int64 v26; // r15
  __int64 v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r15
  __int64 v31; // rax
  __int32 v32; // edx
  __int64 v33; // r8
  __int64 v34; // r9
  __int16 v35; // bx
  unsigned __int16 v36; // bx
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  const __m128i *v39; // rdi
  int v40; // ebx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  const __m128i *v45; // rcx
  int v46; // ebx
  __int64 v47; // rax
  __m128i *v48; // rax
  __m128i *v49; // rbx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int32 v53; // r14d
  __int16 v54; // ax
  __int64 v55; // rcx
  unsigned __int64 v56; // rax
  __int128 v57; // [rsp-20h] [rbp-1E0h]
  __int128 v58; // [rsp-20h] [rbp-1E0h]
  __int32 v59; // [rsp+8h] [rbp-1B8h]
  __int64 v60; // [rsp+18h] [rbp-1A8h]
  int v61; // [rsp+18h] [rbp-1A8h]
  __int16 v62; // [rsp+22h] [rbp-19Eh]
  unsigned __int16 v63; // [rsp+24h] [rbp-19Ch]
  char v64; // [rsp+27h] [rbp-199h]
  __int64 v65; // [rsp+28h] [rbp-198h]
  __int64 v66; // [rsp+28h] [rbp-198h]
  unsigned __int64 v67; // [rsp+30h] [rbp-190h]
  __int64 v68; // [rsp+38h] [rbp-188h]
  unsigned __int8 *v69; // [rsp+48h] [rbp-178h] BYREF
  __m128i v70; // [rsp+50h] [rbp-170h] BYREF
  __int64 v71; // [rsp+60h] [rbp-160h]
  __int64 v72; // [rsp+68h] [rbp-158h]
  __m128i v73; // [rsp+70h] [rbp-150h]
  __m128i v74[2]; // [rsp+80h] [rbp-140h] BYREF
  __int16 v75; // [rsp+A0h] [rbp-120h]
  __int64 v76[6]; // [rsp+D0h] [rbp-F0h] BYREF
  _BYTE *v77; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+108h] [rbp-B8h]
  _BYTE v79[176]; // [rsp+110h] [rbp-B0h] BYREF

  v15 = a11;
  v16 = a9;
  v68 = a12;
  v63 = a11;
  if ( (_WORD)a4 != (_WORD)a11 || (v64 = 0, !(_WORD)a11) && a12 != a5 )
  {
    v64 = a3 & 3;
    if ( !a2 )
      goto LABEL_28;
LABEL_4:
    v65 = a8;
    v17 = (unsigned __int16 *)(*(_QWORD *)(a8 + 48) + 16LL * (unsigned int)a9);
    v18 = sub_33E5B50(a1, a4, a5, *v17, *((_QWORD *)v17 + 1), a6, 1, 0);
    v19 = v65;
    v67 = v18;
    v21 = v20;
    goto LABEL_5;
  }
  if ( a2 )
    goto LABEL_4;
LABEL_28:
  v66 = a8;
  v51 = sub_33E5110(a1, a4, a5, 1, 0);
  v19 = v66;
  v67 = v51;
  v21 = v52;
LABEL_5:
  v22 = _mm_loadu_si128((const __m128i *)&a7);
  v23 = _mm_loadu_si128((const __m128i *)&a10);
  v72 = v16;
  v77 = v79;
  v78 = 0x2000000000LL;
  v60 = v21;
  v71 = v19;
  v70 = v22;
  v73 = v23;
  sub_33C9670((__int64)&v77, 298, v67, (unsigned __int64 *)&v70, 3, v21);
  v24 = v15;
  if ( !v15 )
    v24 = v68;
  v25 = v60;
  v26 = v24;
  v27 = (unsigned int)v24;
  v28 = (unsigned int)v78;
  v29 = (unsigned int)v78 + 1LL;
  if ( v29 > HIDWORD(v78) )
  {
    sub_C8D5F0((__int64)&v77, v79, v29, 4u, v27, v60);
    v28 = (unsigned int)v78;
    v25 = v60;
    v27 = (unsigned int)v26;
  }
  v30 = HIDWORD(v26);
  *(_DWORD *)&v77[4 * v28] = v27;
  LODWORD(v78) = v78 + 1;
  v31 = (unsigned int)v78;
  if ( (unsigned __int64)(unsigned int)v78 + 1 > HIDWORD(v78) )
  {
    v61 = v25;
    sub_C8D5F0((__int64)&v77, v79, (unsigned int)v78 + 1LL, 4u, v27, v25);
    v31 = (unsigned int)v78;
    LODWORD(v25) = v61;
  }
  v59 = v25;
  *(_DWORD *)&v77[4 * v31] = v30;
  *((_QWORD *)&v57 + 1) = v68;
  *(_QWORD *)&v57 = v63;
  v32 = *(_DWORD *)(a6 + 8);
  LODWORD(v78) = v78 + 1;
  v69 = 0;
  sub_33CF750(v74, 298, v32, &v69, v67, v25, v57, (__int64)a13);
  v62 = a2 & 7;
  v35 = v75 & 0xFC7F | (v62 << 7);
  LOBYTE(v75) = v75 & 0x7F | ((_BYTE)v62 << 7);
  HIBYTE(v75) = (4 * v64) | HIBYTE(v35) & 0xF3;
  v36 = v75 & 0xFFFA;
  if ( v76[0] )
    sub_B91220((__int64)v76, v76[0]);
  if ( v69 )
    sub_B91220((__int64)&v69, (__int64)v69);
  v37 = (unsigned int)v78;
  v38 = (unsigned int)v78 + 1LL;
  if ( v38 > HIDWORD(v78) )
  {
    sub_C8D5F0((__int64)&v77, v79, v38, 4u, v33, v34);
    v37 = (unsigned int)v78;
  }
  v39 = a13;
  *(_DWORD *)&v77[4 * v37] = v36;
  LODWORD(v78) = v78 + 1;
  v40 = sub_2EAC1E0((__int64)v39);
  v43 = (unsigned int)v78;
  v44 = (unsigned int)v78 + 1LL;
  if ( v44 > HIDWORD(v78) )
  {
    sub_C8D5F0((__int64)&v77, v79, v44, 4u, v41, v42);
    v43 = (unsigned int)v78;
  }
  v45 = a13;
  *(_DWORD *)&v77[4 * v43] = v40;
  v46 = v45[2].m128i_u16[0];
  LODWORD(v78) = v78 + 1;
  v47 = (unsigned int)v78;
  if ( (unsigned __int64)(unsigned int)v78 + 1 > HIDWORD(v78) )
  {
    sub_C8D5F0((__int64)&v77, v79, (unsigned int)v78 + 1LL, 4u, v41, v42);
    v47 = (unsigned int)v78;
  }
  *(_DWORD *)&v77[4 * v47] = v46;
  LODWORD(v78) = v78 + 1;
  v74[0].m128i_i64[0] = 0;
  v48 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v77, a6, v74[0].m128i_i64);
  v49 = v48;
  if ( v48 )
  {
    sub_2EAC4C0((__m128i *)v48[7].m128i_i64[0], a13);
    goto LABEL_23;
  }
  v49 = (__m128i *)a1[52];
  v53 = *(_DWORD *)(a6 + 8);
  if ( v49 )
  {
    a1[52] = v49->m128i_i64[0];
  }
  else
  {
    v55 = a1[53];
    a1[63] += 120;
    v56 = (v55 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v56 + 120 && v55 )
    {
      a1[53] = v56 + 120;
      if ( !v56 )
        goto LABEL_32;
    }
    else
    {
      v56 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    }
    v49 = (__m128i *)v56;
  }
  *((_QWORD *)&v58 + 1) = v68;
  *(_QWORD *)&v58 = v63;
  sub_33CF750(v49, 298, v53, (unsigned __int8 **)a6, v67, v59, v58, (__int64)a13);
  v54 = v49[2].m128i_i16[0] & 0xFC7F | (v62 << 7);
  v49[2].m128i_i16[0] = v54;
  v49[2].m128i_i8[1] = HIBYTE(v54) & 0xF3 | (4 * v64);
LABEL_32:
  sub_33E4EC0((__int64)a1, (__int64)v49, (__int64)&v70, 3);
  sub_C657C0(a1 + 65, v49->m128i_i64, (__int64 *)v74[0].m128i_i64[0], (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v49);
LABEL_23:
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  return v49;
}
