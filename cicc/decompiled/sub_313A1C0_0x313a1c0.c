// Function: sub_313A1C0
// Address: 0x313a1c0
//
__int64 __fastcall sub_313A1C0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        void (__fastcall *a7)(__int64 *, __int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, __int64, const char *, __int64, void (__fastcall *)(_QWORD, _QWORD, _QWORD)),
        __int64 a8,
        char a9,
        char a10,
        char a11)
{
  _QWORD *v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // r14
  unsigned __int64 v16; // rsi
  int v17; // eax
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rsi
  const char *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int16 v29; // dx
  __int64 v30; // r9
  char v31; // cl
  char v32; // dl
  char v33; // dl
  const char **v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r14
  _QWORD *v37; // rax
  void (__fastcall *v39)(const char **, __int64, __int64); // rax
  unsigned __int64 v41; // rdi
  __m128i *v42; // r14
  unsigned __int64 v43; // rcx
  __int64 v44; // rax
  unsigned __int64 v45; // rsi
  __int64 v46; // rdx
  __m128i *v47; // rax
  __m128i v48; // xmm1
  __int64 v49; // rcx
  __int64 v50; // rdx
  __m128i v51; // xmm0
  void (__fastcall *v52)(const char **, const char **, __int64); // rax
  char v53; // al
  char *v54; // r14
  __int128 v55; // [rsp-10h] [rbp-100h]
  const char **v56; // [rsp+8h] [rbp-E8h]
  __int64 v59; // [rsp+40h] [rbp-B0h]
  _QWORD *v60; // [rsp+48h] [rbp-A8h]
  _QWORD *v62; // [rsp+58h] [rbp-98h]
  __int64 v63; // [rsp+68h] [rbp-88h] BYREF
  __int64 v64; // [rsp+70h] [rbp-80h]
  __int64 v65; // [rsp+78h] [rbp-78h]
  __int64 v66; // [rsp+80h] [rbp-70h]
  const char *v67; // [rsp+90h] [rbp-60h] BYREF
  __int64 v68; // [rsp+98h] [rbp-58h]
  void (__fastcall *v69)(const char **, const char **, __int64); // [rsp+A0h] [rbp-50h]
  __int64 v70; // [rsp+A8h] [rbp-48h]
  unsigned int v71; // [rsp+B0h] [rbp-40h]
  char v72; // [rsp+B4h] [rbp-3Ch]

  if ( a10 )
  {
    v39 = *(void (__fastcall **)(const char **, __int64, __int64))(a6 + 16);
    v69 = 0;
    if ( v39 )
    {
      v39(&v67, a6, 2);
      v70 = *(_QWORD *)(a6 + 24);
      v69 = *(void (__fastcall **)(const char **, const char **, __int64))(a6 + 16);
    }
    v41 = *(unsigned int *)(a2 + 12);
    v72 = a11;
    v42 = (__m128i *)&v67;
    v43 = *(_QWORD *)a2;
    v71 = a3;
    v44 = *(unsigned int *)(a2 + 8);
    v45 = v44 + 1;
    v46 = *(unsigned int *)(a2 + 8);
    if ( v44 + 1 > v41 )
    {
      if ( v43 > (unsigned __int64)&v67 || (unsigned __int64)&v67 >= v43 + 40 * v44 )
      {
        sub_313A0B0(a2, v45, v46, v43, (__int64)a5, a6);
        v44 = *(unsigned int *)(a2 + 8);
        v43 = *(_QWORD *)a2;
        LODWORD(v46) = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v54 = (char *)&v67 - v43;
        sub_313A0B0(a2, v45, v46, v43, (__int64)a5, a6);
        v43 = *(_QWORD *)a2;
        v44 = *(unsigned int *)(a2 + 8);
        v42 = (__m128i *)&v54[*(_QWORD *)a2];
        LODWORD(v46) = *(_DWORD *)(a2 + 8);
      }
    }
    v47 = (__m128i *)(v43 + 40 * v44);
    if ( v47 )
    {
      v48 = _mm_loadu_si128(v47);
      v49 = v47[1].m128i_i64[1];
      v47[1].m128i_i64[0] = 0;
      v50 = v42[1].m128i_i64[0];
      v51 = _mm_loadu_si128(v42);
      v42[1].m128i_i64[0] = 0;
      *v42 = v48;
      v47[1].m128i_i64[0] = v50;
      v46 = v42[1].m128i_i64[1];
      *v47 = v51;
      v47[1].m128i_i64[1] = v46;
      LODWORD(v46) = v42[2].m128i_i32[0];
      v42[1].m128i_i64[1] = v49;
      v47[2].m128i_i32[0] = v46;
      v47[2].m128i_i8[4] = v42[2].m128i_i8[4];
      LODWORD(v46) = *(_DWORD *)(a2 + 8);
    }
    v52 = v69;
    *(_DWORD *)(a2 + 8) = v46 + 1;
    if ( v52 )
      v52(&v67, &v67, 3);
  }
  v13 = *(_QWORD **)(a2 + 560);
  v62 = v13 + 6;
  v14 = v13[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v13 + 6 == (_QWORD *)v14 )
    goto LABEL_22;
  if ( !v14 )
    BUG();
  v60 = (_QWORD *)(v14 - 24);
  if ( *(_BYTE *)(v14 - 24) != 31 )
  {
LABEL_22:
    v36 = *(_QWORD *)(a2 + 584);
    sub_B43C20((__int64)&v67, (__int64)v13);
    v37 = sub_BD2C40(72, unk_3F148B8);
    v60 = v37;
    if ( v37 )
      sub_B4C8A0((__int64)v37, v36, (__int64)v67, v68);
  }
  v67 = "omp_region.end";
  LOWORD(v71) = 259;
  v15 = sub_AA8550(v13, v60 + 3, 0, (__int64)&v67, 0);
  LOWORD(v71) = 259;
  v67 = "omp_region.finalize";
  v16 = v13[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v62 == (_QWORD *)v16 )
  {
    v18 = 0;
  }
  else
  {
    if ( !v16 )
      goto LABEL_47;
    v17 = *(unsigned __int8 *)(v16 - 24);
    v18 = v16 - 24;
    if ( (unsigned int)(v17 - 30) >= 0xB )
      v18 = 0;
  }
  v59 = sub_AA8550(v13, (__int64 *)(v18 + 24), 0, (__int64)&v67, 0);
  v19 = v13[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v62 == (_QWORD *)v19 )
  {
    v21 = 0;
    goto LABEL_13;
  }
  if ( !v19 )
LABEL_47:
    BUG();
  v20 = *(unsigned __int8 *)(v19 - 24);
  v21 = v19 - 24;
  if ( (unsigned int)(v20 - 30) >= 0xB )
    v21 = 0;
LABEL_13:
  sub_D5F1F0(a2 + 512, v21);
  sub_3137750((__int64)&v67, a2, a3, a4, v15, a9);
  v22 = *(const char **)(a2 + 560);
  v23 = *(_QWORD *)(a2 + 568);
  v24 = *(unsigned __int16 *)(a2 + 576);
  LOWORD(v66) = 0;
  v65 = 0;
  v68 = v23;
  LOWORD(v69) = v24;
  v67 = v22;
  v64 = 0;
  a7(&v63, a8, v24, v23, v25, v26, 0, 0, v66, v22, v23, (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v69);
  v27 = v63 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v63 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 24) |= 3u;
    *(_QWORD *)a1 = v27;
  }
  else
  {
    v28 = sub_AA5190(v59);
    if ( v28 )
    {
      v31 = v29;
      v32 = HIBYTE(v29);
    }
    else
    {
      v32 = 0;
      v31 = 0;
    }
    v65 = v28;
    LOBYTE(v66) = v31;
    BYTE1(v66) = v32;
    *((_QWORD *)&v55 + 1) = v28;
    *(_QWORD *)&v55 = v59;
    v64 = v59;
    sub_3138970((__int64)&v67, a2, a3, a5, a10, v30, v55, v66);
    v33 = v70 & 1;
    LOBYTE(v70) = (2 * (v70 & 1)) | v70 & 0xFD;
    if ( v33 )
    {
      v34 = &v67;
      sub_3139B90(&v63, (__int64 *)&v67);
      v35 = v63;
      *(_BYTE *)(a1 + 24) |= 3u;
      *(_QWORD *)a1 = v35 & 0xFFFFFFFFFFFFFFFELL;
    }
    else
    {
      sub_F39690(v59, 0, 0, 0, 0, 0, 0);
      v34 = v56;
      if ( (unsigned __int8)sub_F39690(v15, 0, 0, 0, 0, 0, 0) )
        v15 = v60[5];
      if ( *(_BYTE *)v60 != 31 )
        sub_B43D60(v60);
      *(_QWORD *)(a2 + 560) = v15;
      *(_QWORD *)(a2 + 568) = v15 + 48;
      *(_WORD *)(a2 + 576) = 0;
      v53 = *(_BYTE *)(a1 + 24);
      *(_QWORD *)(a1 + 8) = v15 + 48;
      *(_QWORD *)a1 = v15;
      *(_WORD *)(a1 + 16) = 0;
      *(_BYTE *)(a1 + 24) = v53 & 0xFC | 2;
    }
    if ( (v70 & 2) != 0 )
      sub_267DA20(&v67, (__int64)v34);
    if ( (v70 & 1) != 0 && v67 )
      (*(void (__fastcall **)(const char *))(*(_QWORD *)v67 + 8LL))(v67);
  }
  return a1;
}
