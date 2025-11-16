// Function: sub_1FEBFF0
// Address: 0x1febff0
//
__int64 __fastcall sub_1FEBFF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8)
{
  __int64 v13; // rax
  char v14; // di
  __int64 v15; // rax
  unsigned int v16; // r10d
  __m128i v17; // xmm0
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rcx
  const void **v23; // r8
  __int64 v24; // rax
  char v25; // r10
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 result; // rax
  const void **v31; // rdx
  __int64 v32; // r9
  _QWORD *v33; // rax
  unsigned __int64 v34; // rdx
  unsigned int v35; // edi
  __m128i v36; // xmm1
  _QWORD *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r10
  __int64 v40; // r11
  int v41; // edx
  __m128i v42; // xmm2
  _QWORD *v43; // rdi
  int v44; // edx
  __int64 v45; // rdi
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  _QWORD *v48; // r11
  unsigned int v49; // r10d
  __int64 v50; // r13
  __int64 v51; // r12
  __int128 v52; // rax
  __int64 *v53; // rax
  int v54; // edx
  __m128i v55; // xmm3
  __int128 v56; // [rsp-60h] [rbp-160h]
  __int64 v57; // [rsp-40h] [rbp-140h]
  __int128 v58; // [rsp-40h] [rbp-140h]
  __int64 v59; // [rsp-38h] [rbp-138h]
  __int128 v60; // [rsp-30h] [rbp-130h]
  __int64 v61; // [rsp-30h] [rbp-130h]
  __int64 v62; // [rsp-20h] [rbp-120h]
  __int128 v63; // [rsp-10h] [rbp-110h]
  __int64 v64; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v65; // [rsp+18h] [rbp-E8h]
  __int64 v66; // [rsp+20h] [rbp-E0h]
  _QWORD *v67; // [rsp+28h] [rbp-D8h]
  unsigned int v68; // [rsp+30h] [rbp-D0h]
  _QWORD *v69; // [rsp+30h] [rbp-D0h]
  __int64 *v70; // [rsp+30h] [rbp-D0h]
  unsigned int v71; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v72; // [rsp+3Fh] [rbp-C1h]
  _BYTE *v73; // [rsp+40h] [rbp-C0h]
  __int64 *v74; // [rsp+40h] [rbp-C0h]
  unsigned int v75; // [rsp+48h] [rbp-B8h]
  char v76; // [rsp+48h] [rbp-B8h]
  unsigned int v77; // [rsp+48h] [rbp-B8h]
  __m128i v78; // [rsp+70h] [rbp-90h] BYREF
  __int64 v79; // [rsp+80h] [rbp-80h]
  __int64 v80; // [rsp+90h] [rbp-70h]
  __int64 v81; // [rsp+98h] [rbp-68h]
  __m128i v82; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v83; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-48h]
  __int64 v85; // [rsp+C0h] [rbp-40h]

  v13 = *(_QWORD *)(a4 + 40) + 16LL * (unsigned int)a5;
  v14 = *(_BYTE *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v82.m128i_i8[0] = v14;
  v82.m128i_i64[1] = v15;
  if ( v14 )
    v16 = sub_1FEB8F0(v14);
  else
    v16 = sub_1F58D40((__int64)&v82);
  v17 = _mm_loadu_si128(&v82);
  *(__m128i *)a2 = v17;
  v18 = *(_QWORD *)(a1 + 16);
  if ( v16 == 32 )
  {
    v19 = 5;
    LOBYTE(v20) = 5;
    goto LABEL_7;
  }
  if ( v16 <= 0x20 )
  {
    if ( v16 == 8 )
    {
      v19 = 3;
      LOBYTE(v20) = 3;
    }
    else
    {
      v19 = 4;
      LOBYTE(v20) = 4;
      if ( v16 != 16 )
      {
        v19 = 2;
        LOBYTE(v20) = 2;
        if ( v16 != 1 )
          goto LABEL_17;
      }
    }
LABEL_7:
    v21 = *(_QWORD *)(a1 + 8);
    v22 = (unsigned __int8)v20;
    v23 = 0;
    goto LABEL_8;
  }
  if ( v16 == 64 )
  {
    v19 = 6;
    LOBYTE(v20) = 6;
    goto LABEL_7;
  }
  if ( v16 == 128 )
  {
    v19 = 7;
    LOBYTE(v20) = 7;
    goto LABEL_7;
  }
LABEL_17:
  v77 = v16;
  v20 = sub_1F58CC0(*(_QWORD **)(v18 + 48), v16);
  v16 = v77;
  v22 = v20;
  v23 = v31;
  if ( !(_BYTE)v20 )
  {
    v18 = *(_QWORD *)(a1 + 16);
LABEL_19:
    v73 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(v18 + 32));
    v72 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1158LL);
    v33 = sub_1D29D50(*(_QWORD **)(a1 + 16), v82.m128i_u32[0], v82.m128i_i64[1], v72, 0, v32);
    v35 = v34;
    v64 = (__int64)v33;
    v65 = v34;
    LODWORD(v34) = *((_DWORD *)v33 + 21);
    *(_DWORD *)(a2 + 40) = v35;
    *(_QWORD *)(a2 + 32) = v33;
    v67 = v33;
    v68 = v35;
    v71 = v34;
    v66 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
    sub_1E341E0((__int64)&v78, v66, v34, 0);
    v36 = _mm_loadu_si128(&v78);
    *(_QWORD *)(a2 + 104) = v79;
    *(__m128i *)(a2 + 88) = v36;
    v37 = *(_QWORD **)(a1 + 16);
    v62 = *(_QWORD *)(a2 + 104);
    v60 = *(_OWORD *)(a2 + 88);
    v59 = *(_QWORD *)(a2 + 40);
    v57 = *(_QWORD *)(a2 + 32);
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v38 = sub_1D2BF40(v37, (__int64)(v37 + 11), 0, a3, a4, a5, v57, v59, v60, v62, 0, 0, (__int64)&v83);
    v39 = v68;
    v40 = (__int64)v67;
    *(_QWORD *)(a2 + 16) = v38;
    *(_DWORD *)(a2 + 24) = v41;
    if ( *v73 )
    {
      v42 = _mm_loadu_si128((const __m128i *)(a2 + 88));
      *(_QWORD *)(a2 + 80) = *(_QWORD *)(a2 + 104);
      *(__m128i *)(a2 + 64) = v42;
    }
    else
    {
      if ( v82.m128i_i8[0] )
      {
        v47 = sub_1FEB8F0(v82.m128i_i8[0]);
      }
      else
      {
        v47 = sub_1F58D40((__int64)&v82);
        v48 = v67;
        v49 = v68;
      }
      v50 = 16LL * v49;
      v51 = (v47 >> 3) - 1;
      v69 = v48;
      v74 = *(__int64 **)(a1 + 16);
      *(_QWORD *)&v52 = sub_1D38BB0(
                          (__int64)v74,
                          v51,
                          a3,
                          *(unsigned __int8 *)(v50 + v48[5]),
                          *(const void ***)(v50 + v48[5] + 8),
                          0,
                          v17,
                          *(double *)v36.m128i_i64,
                          a8,
                          0);
      v53 = sub_1D332F0(
              v74,
              52,
              a3,
              *(unsigned __int8 *)(v69[5] + v50),
              *(const void ***)(v69[5] + v50 + 8),
              0,
              *(double *)v17.m128i_i64,
              *(double *)v36.m128i_i64,
              a8,
              v64,
              v65,
              v52);
      LODWORD(v74) = v54;
      v70 = v53;
      sub_1E341E0((__int64)&v78, v66, v71, v51);
      v55 = _mm_loadu_si128(&v78);
      v40 = (__int64)v70;
      v39 = (unsigned int)v74;
      *(_QWORD *)(a2 + 80) = v79;
      *(__m128i *)(a2 + 64) = v55;
    }
    *(_QWORD *)(a2 + 48) = v40;
    *(_DWORD *)(a2 + 56) = v39;
    v43 = *(_QWORD **)(a1 + 16);
    v61 = *(_QWORD *)(a2 + 80);
    v58 = *(_OWORD *)(a2 + 64);
    v56 = *(_OWORD *)(a2 + 16);
    v83 = 0;
    v84 = 0;
    v85 = 0;
    *(_QWORD *)(a2 + 112) = sub_1D2B810(v43, 1u, a3, v72, 0, 0, v56, v40, v39, v58, v61, 3, 0, 0, (__int64)&v83);
    *(_DWORD *)(a2 + 120) = v44;
    LODWORD(v84) = sub_1FEB8F0(v72);
    if ( (unsigned int)v84 > 0x40 )
    {
      sub_16A4EF0((__int64)&v83, 0, 0);
      if ( (unsigned int)v84 > 0x40 )
      {
        *(_QWORD *)v83 |= 0x80uLL;
LABEL_24:
        if ( *(_DWORD *)(a2 + 136) > 0x40u )
        {
          v45 = *(_QWORD *)(a2 + 128);
          if ( v45 )
            j_j___libc_free_0_0(v45);
        }
        v46 = v83;
        *(_BYTE *)(a2 + 144) = 7;
        *(_QWORD *)(a2 + 128) = v46;
        result = (unsigned int)v84;
        *(_DWORD *)(a2 + 136) = v84;
        return result;
      }
    }
    else
    {
      v83 = 0;
    }
    v83 |= 0x80u;
    goto LABEL_24;
  }
  v21 = *(_QWORD *)(a1 + 8);
  v18 = *(_QWORD *)(a1 + 16);
  v19 = (unsigned __int8)v20;
LABEL_8:
  v75 = v16;
  if ( !*(_QWORD *)(v21 + 8 * v19 + 120) )
    goto LABEL_19;
  *((_QWORD *)&v63 + 1) = a5;
  LOBYTE(v22) = v20;
  *(_QWORD *)&v63 = a4;
  v24 = sub_1D309E0((__int64 *)v18, 158, a3, v22, v23, 0, *(double *)v17.m128i_i64, a7, *(double *)a8.m128i_i64, v63);
  v25 = v75;
  v81 = v26;
  v80 = v24;
  *(_QWORD *)(a2 + 112) = v24;
  LODWORD(v84) = v75;
  v27 = 1LL << ((unsigned __int8)v75 - 1);
  *(_DWORD *)(a2 + 120) = v81;
  if ( v75 <= 0x40 )
  {
    v83 = 0;
    goto LABEL_40;
  }
  sub_16A4EF0((__int64)&v83, 0, 0);
  v25 = v75;
  if ( (unsigned int)v84 <= 0x40 )
  {
LABEL_40:
    v83 |= v27;
    goto LABEL_12;
  }
  *(_QWORD *)(v83 + 8LL * ((v75 - 1) >> 6)) |= v27;
LABEL_12:
  if ( *(_DWORD *)(a2 + 136) > 0x40u )
  {
    v28 = *(_QWORD *)(a2 + 128);
    if ( v28 )
    {
      v76 = v25;
      j_j___libc_free_0_0(v28);
      v25 = v76;
    }
  }
  v29 = v83;
  *(_BYTE *)(a2 + 144) = v25 - 1;
  *(_QWORD *)(a2 + 128) = v29;
  result = (unsigned int)v84;
  *(_DWORD *)(a2 + 136) = v84;
  return result;
}
