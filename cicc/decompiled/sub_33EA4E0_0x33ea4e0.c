// Function: sub_33EA4E0
// Address: 0x33ea4e0
//
_QWORD *__fastcall sub_33EA4E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        char a8)
{
  unsigned __int16 *v12; // rax
  __int64 v13; // rax
  __m128i v14; // xmm2
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int32 v17; // edx
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int64 v20; // r9
  __int64 v21; // r9
  unsigned __int64 v22; // r8
  __int64 v23; // rax
  int v24; // r11d
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned __int16 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rdi
  int v31; // eax
  __int64 v32; // r9
  __int64 v33; // rdx
  unsigned __int64 v34; // r8
  __int64 v35; // rax
  __int64 v36; // r8
  _QWORD *v37; // rax
  _QWORD *v38; // r13
  __int64 v40; // r8
  unsigned __int16 v41; // r9
  __int64 v42; // r11
  char v43; // bl
  __m128i *v44; // rax
  __int32 v45; // r10d
  __int64 v46; // r8
  __int64 v47; // rsi
  unsigned __int64 v48; // rdx
  __int128 v49; // [rsp-20h] [rbp-170h]
  __int64 v50; // [rsp-10h] [rbp-160h]
  __int64 v51; // [rsp+8h] [rbp-148h]
  __int64 v52; // [rsp+10h] [rbp-140h]
  int v53; // [rsp+1Ch] [rbp-134h]
  __int32 v54; // [rsp+1Ch] [rbp-134h]
  unsigned __int64 v55; // [rsp+20h] [rbp-130h]
  int v56; // [rsp+20h] [rbp-130h]
  int v57; // [rsp+20h] [rbp-130h]
  int v58; // [rsp+20h] [rbp-130h]
  int v59; // [rsp+20h] [rbp-130h]
  unsigned __int16 v60; // [rsp+20h] [rbp-130h]
  __int32 v61; // [rsp+28h] [rbp-128h]
  __int64 v62; // [rsp+30h] [rbp-120h]
  __m128i *v63; // [rsp+30h] [rbp-120h]
  __int64 *v64; // [rsp+38h] [rbp-118h]
  __int64 *v65; // [rsp+48h] [rbp-108h] BYREF
  _OWORD v66[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v67; // [rsp+70h] [rbp-E0h]
  __int64 v68; // [rsp+78h] [rbp-D8h]
  __m128i v69; // [rsp+80h] [rbp-D0h]
  _BYTE *v70; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+98h] [rbp-B8h]
  _BYTE v72[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v12 = (unsigned __int16 *)(*(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6);
  v13 = sub_33E5110(a1, *v12, *((_QWORD *)v12 + 1), 1, 0);
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v15 = v13;
  v62 = v13;
  v16 = *(_QWORD *)(a2 + 40);
  v61 = v17;
  v18 = _mm_loadu_si128((const __m128i *)v16);
  v19 = _mm_loadu_si128((const __m128i *)(v16 + 40));
  v68 = a6;
  v67 = a5;
  v71 = 0x2000000000LL;
  v70 = v72;
  v66[0] = v18;
  v66[1] = v19;
  v69 = v14;
  sub_33C9670((__int64)&v70, 299, v15, (unsigned __int64 *)v66, 4, v20);
  v22 = *(unsigned __int16 *)(a2 + 96);
  v23 = (unsigned int)v71;
  if ( !(_WORD)v22 )
    v22 = *(_QWORD *)(a2 + 104);
  v24 = v22;
  if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
  {
    v53 = v22;
    v55 = v22;
    sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 4u, v22, v21);
    v23 = (unsigned int)v71;
    v24 = v53;
    v22 = v55;
  }
  v25 = HIDWORD(v22);
  *(_DWORD *)&v70[4 * v23] = v24;
  LODWORD(v71) = v71 + 1;
  v26 = (unsigned int)v71;
  if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
  {
    v56 = v25;
    sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 4u, v25, v21);
    v26 = (unsigned int)v71;
    LODWORD(v25) = v56;
  }
  *(_DWORD *)&v70[4 * v26] = v25;
  v27 = *(_WORD *)(a2 + 32);
  LODWORD(v71) = v71 + 1;
  v28 = (unsigned int)v71;
  v29 = v27 & 0xFFFA;
  if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
  {
    v57 = v29;
    sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 4u, v29, v21);
    v28 = (unsigned int)v71;
    LODWORD(v29) = v57;
  }
  *(_DWORD *)&v70[4 * v28] = v29;
  v30 = *(_QWORD *)(a2 + 112);
  LODWORD(v71) = v71 + 1;
  v31 = sub_2EAC1E0(v30);
  v33 = (unsigned int)v71;
  v34 = (unsigned int)v71 + 1LL;
  if ( v34 > HIDWORD(v71) )
  {
    v58 = v31;
    sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 4u, v34, v32);
    v33 = (unsigned int)v71;
    v31 = v58;
  }
  *(_DWORD *)&v70[4 * v33] = v31;
  v36 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 112) + 32LL);
  LODWORD(v71) = v71 + 1;
  v35 = (unsigned int)v71;
  if ( (unsigned __int64)(unsigned int)v71 + 1 > HIDWORD(v71) )
  {
    v59 = v36;
    sub_C8D5F0((__int64)&v70, v72, (unsigned int)v71 + 1LL, 4u, v36, v32);
    v35 = (unsigned int)v71;
    LODWORD(v36) = v59;
  }
  *(_DWORD *)&v70[4 * v35] = v36;
  LODWORD(v71) = v71 + 1;
  v65 = 0;
  v37 = sub_33CCCF0((__int64)a1, (__int64)&v70, a4, (__int64 *)&v65);
  if ( !v37 )
  {
    v40 = *(_QWORD *)(a2 + 112);
    v41 = *(_WORD *)(a2 + 96);
    v42 = *(_QWORD *)(a2 + 104);
    v43 = *(_BYTE *)(a2 + 33);
    v44 = (__m128i *)a1[52];
    v45 = *(_DWORD *)(a4 + 8);
    if ( v44 )
    {
      a1[52] = v44->m128i_i64[0];
    }
    else
    {
      v47 = a1[53];
      a1[63] += 120;
      v48 = (v47 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v48 + 120 && v47 )
      {
        a1[53] = v48 + 120;
        if ( !v48 )
          goto LABEL_20;
        v44 = (__m128i *)((v47 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        v51 = v42;
        v52 = v40;
        v54 = v45;
        v60 = v41;
        v44 = (__m128i *)sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v42 = v51;
        v40 = v52;
        v45 = v54;
        v41 = v60;
      }
    }
    v50 = v40;
    *((_QWORD *)&v49 + 1) = v42;
    v46 = v62;
    *(_QWORD *)&v49 = v41;
    v63 = v44;
    sub_33CF750(v44, 299, v45, (unsigned __int8 **)a4, v46, v61, v49, v50);
    v44 = v63;
    v63[2].m128i_i16[0] = v63[2].m128i_i16[0] & 0xF87F | ((a8 & 7) << 7) | (((v43 & 4) != 0) << 10);
LABEL_20:
    v64 = (__int64 *)v44;
    sub_33E4EC0((__int64)a1, (__int64)v44, (__int64)v66, 4);
    sub_C657C0(a1 + 65, v64, v65, (__int64)off_4A367D0);
    sub_33CC420((__int64)a1, (__int64)v64);
    v37 = v64;
  }
  v38 = v37;
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  return v38;
}
