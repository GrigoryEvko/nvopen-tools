// Function: sub_33F49B0
// Address: 0x33f49b0
//
__m128i *__fastcall sub_33F49B0(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        __int64 a9,
        unsigned __int64 a10,
        const __m128i *a11)
{
  __int64 v12; // rax
  unsigned __int16 *v14; // rax
  __int32 v15; // edx
  __int64 v16; // r8
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  int v19; // edx
  unsigned __int64 v20; // r11
  unsigned __int64 v21; // r10
  __int64 v22; // r9
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // r14
  __int64 v26; // r8
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r9
  int v30; // eax
  __int64 v31; // rdx
  unsigned __int64 v32; // r8
  int v33; // eax
  __int64 v34; // r9
  __int64 v35; // rdx
  unsigned __int64 v36; // r8
  __int64 v37; // r8
  __int64 v38; // rax
  __m128i *v39; // rax
  __m128i *v40; // r13
  __int32 v41; // r14d
  __int16 v42; // ax
  __int64 v43; // rcx
  unsigned __int64 v44; // rax
  __int128 v45; // [rsp-20h] [rbp-210h]
  __int128 v46; // [rsp-20h] [rbp-210h]
  unsigned __int64 v47; // [rsp+8h] [rbp-1E8h]
  int v48; // [rsp+10h] [rbp-1E0h]
  _QWORD *v50; // [rsp+18h] [rbp-1D8h]
  __int64 v51; // [rsp+20h] [rbp-1D0h]
  __int32 v52; // [rsp+28h] [rbp-1C8h]
  __m128i *v53; // [rsp+38h] [rbp-1B8h]
  __int16 v55; // [rsp+40h] [rbp-1B0h]
  __int16 v56; // [rsp+40h] [rbp-1B0h]
  int v57; // [rsp+40h] [rbp-1B0h]
  int v58; // [rsp+40h] [rbp-1B0h]
  int v59; // [rsp+40h] [rbp-1B0h]
  unsigned __int8 *v61; // [rsp+68h] [rbp-188h] BYREF
  unsigned __int64 v62[7]; // [rsp+70h] [rbp-180h] BYREF
  int v63; // [rsp+A8h] [rbp-148h]
  __m128i v64[2]; // [rsp+B0h] [rbp-140h] BYREF
  __int16 v65; // [rsp+D0h] [rbp-120h]
  __int64 v66[6]; // [rsp+100h] [rbp-F0h] BYREF
  _BYTE *v67; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+138h] [rbp-B8h]
  _BYTE v69[176]; // [rsp+140h] [rbp-B0h] BYREF

  v12 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  if ( *(_WORD *)v12 == (_WORD)a9 && (*(_QWORD *)(v12 + 8) == a10 || (_WORD)a9) )
    return sub_33F3F90(a1, a2, a3, a4, a5, a6, a7, a8, a11);
  v53 = sub_33ED250((__int64)a1, 1, 0);
  v14 = (unsigned __int16 *)(*(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8);
  v52 = v15;
  v16 = *((_QWORD *)v14 + 1);
  v17 = *v14;
  v67 = 0;
  LODWORD(v68) = 0;
  v18 = sub_33F17F0(a1, 51, (__int64)&v67, v17, v16);
  v20 = a2;
  v21 = a5;
  v22 = a9;
  if ( v67 )
  {
    v47 = a5;
    v48 = v19;
    v50 = v18;
    sub_B91220((__int64)&v67, (__int64)v67);
    v22 = a9;
    v21 = v47;
    v19 = v48;
    v18 = v50;
    v20 = a2;
  }
  v63 = v19;
  v62[4] = a7;
  v62[1] = a3;
  v62[3] = a6;
  v62[6] = (unsigned __int64)v18;
  v51 = v22;
  v62[5] = a8;
  v68 = 0x2000000000LL;
  v62[0] = v20;
  v62[2] = v21;
  v67 = v69;
  sub_33C9670((__int64)&v67, 299, (unsigned __int64)v53, v62, 4, v22);
  v23 = v51;
  v24 = (unsigned int)v68;
  v25 = (unsigned __int16)v51;
  if ( !(_WORD)v51 )
    v25 = a10;
  v26 = (unsigned int)v25;
  if ( (unsigned __int64)(unsigned int)v68 + 1 > HIDWORD(v68) )
  {
    sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 4u, (unsigned int)v25, v51);
    v24 = (unsigned int)v68;
    v26 = (unsigned int)v25;
  }
  v27 = HIDWORD(v25);
  *(_DWORD *)&v67[4 * v24] = v26;
  LODWORD(v68) = v68 + 1;
  v28 = (unsigned int)v68;
  if ( (unsigned __int64)(unsigned int)v68 + 1 > HIDWORD(v68) )
  {
    sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 4u, v26, v23);
    v28 = (unsigned int)v68;
  }
  *(_DWORD *)&v67[4 * v28] = v27;
  LODWORD(v68) = v68 + 1;
  v61 = 0;
  *((_QWORD *)&v45 + 1) = a10;
  *(_QWORD *)&v45 = (unsigned __int16)a9;
  sub_33CF750(v64, 299, *(_DWORD *)(a4 + 8), &v61, (__int64)v53, v52, v45, (__int64)a11);
  LOBYTE(v30) = v65 & 0x7F;
  BYTE1(v30) = ((unsigned __int16)(v65 & 0xF87F) >> 8) | 4;
  v65 = v30;
  LOBYTE(v30) = v30 & 0x7A;
  if ( v66[0] )
  {
    v55 = v30;
    sub_B91220((__int64)v66, v66[0]);
    LOWORD(v30) = v55;
  }
  if ( v61 )
  {
    v56 = v30;
    sub_B91220((__int64)&v61, (__int64)v61);
    LOWORD(v30) = v56;
  }
  v31 = (unsigned int)v68;
  v30 = (unsigned __int16)v30;
  v32 = (unsigned int)v68 + 1LL;
  if ( v32 > HIDWORD(v68) )
  {
    v57 = (unsigned __int16)v30;
    sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 4u, v32, v29);
    v31 = (unsigned int)v68;
    v30 = v57;
  }
  *(_DWORD *)&v67[4 * v31] = v30;
  LODWORD(v68) = v68 + 1;
  v33 = sub_2EAC1E0((__int64)a11);
  v35 = (unsigned int)v68;
  v36 = (unsigned int)v68 + 1LL;
  if ( v36 > HIDWORD(v68) )
  {
    v58 = v33;
    sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 4u, v36, v34);
    v35 = (unsigned int)v68;
    v33 = v58;
  }
  *(_DWORD *)&v67[4 * v35] = v33;
  v37 = a11[2].m128i_u16[0];
  LODWORD(v68) = v68 + 1;
  v38 = (unsigned int)v68;
  if ( (unsigned __int64)(unsigned int)v68 + 1 > HIDWORD(v68) )
  {
    v59 = v37;
    sub_C8D5F0((__int64)&v67, v69, (unsigned int)v68 + 1LL, 4u, v37, v34);
    v38 = (unsigned int)v68;
    LODWORD(v37) = v59;
  }
  *(_DWORD *)&v67[4 * v38] = v37;
  LODWORD(v68) = v68 + 1;
  v64[0].m128i_i64[0] = 0;
  v39 = (__m128i *)sub_33CCCF0((__int64)a1, (__int64)&v67, a4, v64[0].m128i_i64);
  v40 = v39;
  if ( !v39 )
  {
    v40 = (__m128i *)a1[52];
    v41 = *(_DWORD *)(a4 + 8);
    if ( v40 )
    {
      a1[52] = v40->m128i_i64[0];
    }
    else
    {
      v43 = a1[53];
      a1[63] += 120LL;
      v44 = (v43 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v44 + 120 && v43 )
      {
        a1[53] = v44 + 120;
        if ( !v44 )
          goto LABEL_31;
      }
      else
      {
        v44 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      }
      v40 = (__m128i *)v44;
    }
    *((_QWORD *)&v46 + 1) = a10;
    *(_QWORD *)&v46 = (unsigned __int16)a9;
    sub_33CF750(v40, 299, v41, (unsigned __int8 **)a4, (__int64)v53, v52, v46, (__int64)a11);
    v42 = v40[2].m128i_i16[0] & 0xF87F;
    HIBYTE(v42) |= 4u;
    v40[2].m128i_i16[0] = v42;
LABEL_31:
    sub_33E4EC0((__int64)a1, (__int64)v40, (__int64)v62, 4);
    sub_C657C0(a1 + 65, v40->m128i_i64, (__int64 *)v64[0].m128i_i64[0], (__int64)off_4A367D0);
    sub_33CC420((__int64)a1, (__int64)v40);
    goto LABEL_25;
  }
  sub_2EAC4C0((__m128i *)v39[7].m128i_i64[0], a11);
LABEL_25:
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  return v40;
}
