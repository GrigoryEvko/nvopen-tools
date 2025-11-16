// Function: sub_3459BD0
// Address: 0x3459bd0
//
unsigned __int8 *__fastcall sub_3459BD0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r14
  __int64 v6; // rsi
  __int128 v7; // xmm0
  __m128i v8; // xmm2
  unsigned __int16 *v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // cl
  unsigned __int64 v13; // rsi
  __int64 *v14; // rdi
  __int16 v15; // bx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r15d
  __int128 v19; // rax
  int v20; // r9d
  __int128 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int128 v24; // rax
  __int128 v25; // rax
  __int64 v26; // r9
  unsigned __int8 *v27; // r14
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rax
  char v35; // cl
  __int64 *v36; // rdi
  __int16 v37; // ax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rcx
  __int128 v41; // rax
  __int64 v42; // r9
  unsigned int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int128 v47; // [rsp+0h] [rbp-100h]
  __int128 v48; // [rsp+0h] [rbp-100h]
  __int64 v49; // [rsp+18h] [rbp-E8h]
  __int128 v50; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v51; // [rsp+30h] [rbp-D0h]
  __int128 v52; // [rsp+30h] [rbp-D0h]
  unsigned __int16 v53; // [rsp+40h] [rbp-C0h]
  __int128 v54; // [rsp+40h] [rbp-C0h]
  __int64 v55; // [rsp+50h] [rbp-B0h]
  __int64 v56; // [rsp+58h] [rbp-A8h]
  __int64 v57; // [rsp+90h] [rbp-70h] BYREF
  int v58; // [rsp+98h] [rbp-68h]
  unsigned int v59; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v60; // [rsp+A8h] [rbp-58h]
  __int64 v61; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-48h]
  __int64 v63; // [rsp+C0h] [rbp-40h] BYREF
  int v64; // [rsp+C8h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 80);
  v57 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v57, v6, 1);
  v58 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(a2 + 40);
  v7 = (__int128)_mm_loadu_si128((const __m128i *)v10);
  v8 = _mm_loadu_si128((const __m128i *)(v10 + 80));
  v51 = *(unsigned __int8 **)v10;
  v49 = *(unsigned int *)(v10 + 8);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16 * v49);
  v50 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  LODWORD(v10) = *v9;
  v60 = *((_QWORD *)v9 + 1);
  v11 = *(_QWORD *)(a2 + 48);
  LOWORD(v59) = v10;
  v53 = *(_WORD *)v11;
  v55 = *(_QWORD *)(v11 + 8);
  if ( (_WORD)v10 )
  {
    v12 = (unsigned __int16)(v10 - 176) <= 0x34u;
    v13 = word_4456340[(int)v10 - 1];
    LOBYTE(v10) = v12;
  }
  else
  {
    v13 = sub_3007240((__int64)&v59);
    v10 = HIDWORD(v13);
    v12 = BYTE4(v13);
  }
  v14 = (__int64 *)a3[8];
  LODWORD(v63) = v13;
  BYTE4(v63) = v10;
  if ( v12 )
  {
    v56 = 0;
    v15 = sub_2D43AD0(v53, v13);
    if ( v15 )
      goto LABEL_7;
  }
  else
  {
    v56 = 0;
    v15 = sub_2D43050(v53, v13);
    if ( v15 )
      goto LABEL_7;
  }
  v13 = v53;
  v44 = sub_3009450(v14, v53, v55, v63, v16, v17);
  v56 = v45;
  v3 = v44;
  v15 = v44;
LABEL_7:
  LOWORD(v3) = v15;
  if ( (_WORD)v59 )
  {
    if ( (unsigned __int16)(v59 - 17) > 0xD3u )
    {
      if ( (_WORD)v59 == 2 )
        goto LABEL_10;
LABEL_20:
      *(_QWORD *)&v52 = sub_3400BD0((__int64)a3, 0, (__int64)&v57, v59, v60, 0, (__m128i)v7, 0);
      *((_QWORD *)&v52 + 1) = v32;
      if ( (_WORD)v59 )
      {
        v35 = (unsigned __int16)(v59 - 176) <= 0x34u;
        LODWORD(v33) = word_4456340[(unsigned __int16)v59 - 1];
        LOBYTE(v34) = v35;
      }
      else
      {
        v33 = sub_3007240((__int64)&v59);
        v34 = HIDWORD(v33);
        v35 = BYTE4(v33);
      }
      v36 = (__int64 *)a3[8];
      LODWORD(v63) = v33;
      BYTE4(v63) = v34;
      if ( v35 )
      {
        v37 = sub_2D43AD0(2, v33);
        v40 = 0;
        if ( v37 )
        {
LABEL_24:
          v60 = v40;
          LOWORD(v59) = v37;
          *(_QWORD *)&v41 = sub_33ED040(a3, 0x16u);
          v51 = sub_33FC1D0(a3, 463, (__int64)&v57, v59, v60, v42, v7, v52, v41, v50, *(_OWORD *)&v8);
          v49 = v43;
          goto LABEL_10;
        }
      }
      else
      {
        v37 = sub_2D43050(2, v33);
        v40 = 0;
        if ( v37 )
          goto LABEL_24;
      }
      v37 = sub_3009450(v36, 2, 0, v63, v38, v39);
      v40 = v46;
      goto LABEL_24;
    }
    if ( word_4456580[(unsigned __int16)v59 - 1] != 2 )
      goto LABEL_20;
  }
  else if ( !sub_30070B0((__int64)&v59) || (unsigned __int16)sub_3009970((__int64)&v59, v13, v29, v30, v31) != 2 )
  {
    goto LABEL_20;
  }
LABEL_10:
  v18 = v53;
  *(_QWORD *)&v19 = sub_33FB310((__int64)a3, v8.m128i_i64[0], v8.m128i_u32[2], (__int64)&v57, v53, v55, (__m128i)v7);
  v61 = v3;
  v54 = v19;
  v62 = v56;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 176) > 0x34u )
    {
LABEL_12:
      *(_QWORD *)&v21 = sub_32886A0((__int64)a3, (unsigned int)v61, v62, (int)&v57, v54, SDWORD2(v54));
      goto LABEL_13;
    }
  }
  else if ( !sub_3007100((__int64)&v61) )
  {
    goto LABEL_12;
  }
  if ( *(_DWORD *)(v54 + 24) == 51 )
  {
    v63 = 0;
    v64 = 0;
    *(_QWORD *)&v21 = sub_33F17F0(a3, 51, (__int64)&v63, v61, v62);
    if ( v63 )
    {
      v48 = v21;
      sub_B91220((__int64)&v63, v63);
      v21 = v48;
    }
  }
  else
  {
    *(_QWORD *)&v21 = sub_33FAF80((__int64)a3, 168, (__int64)&v57, v61, v62, v20, (__m128i)v7);
  }
LABEL_13:
  v47 = v21;
  *(_QWORD *)&v24 = sub_3402A00(a3, (unsigned __int64 *)&v57, (unsigned int)v3, v56, (__m128i)v7, v22, v23);
  *(_QWORD *)&v25 = sub_33FC130(
                      a3,
                      488,
                      (__int64)&v57,
                      (unsigned int)v3,
                      v56,
                      *((__int64 *)&v47 + 1),
                      __PAIR128__(v49 | *((_QWORD *)&v7 + 1) & 0xFFFFFFFF00000000LL, (unsigned __int64)v51),
                      v24,
                      v47,
                      *(_OWORD *)&v8);
  v27 = sub_33FC130(a3, 479, (__int64)&v57, v18, v55, v26, v54, v25, v50, *(_OWORD *)&v8);
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
  return v27;
}
