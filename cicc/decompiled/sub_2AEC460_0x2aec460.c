// Function: sub_2AEC460
// Address: 0x2aec460
//
char *__fastcall sub_2AEC460(__int64 a1, int a2, unsigned __int64 a3, char a4)
{
  unsigned int v5; // r13d
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  int v8; // eax
  char *v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rsi
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char v20; // al
  __int64 *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rsi
  __int64 v27; // rax
  __int8 *v28; // rsi
  size_t v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  int v34; // r13d
  char v35; // bl
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdi
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // r13
  __int64 v42; // r13
  __int64 v43; // rbx
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // [rsp+0h] [rbp-4B0h]
  __int64 v54; // [rsp+0h] [rbp-4B0h]
  unsigned __int64 v57; // [rsp+28h] [rbp-488h]
  __int64 *v58; // [rsp+28h] [rbp-488h]
  __int64 v60; // [rsp+58h] [rbp-458h] BYREF
  __int64 v61; // [rsp+60h] [rbp-450h]
  __int64 v62; // [rsp+68h] [rbp-448h]
  __m128i v63; // [rsp+70h] [rbp-440h] BYREF
  __m128i v64[2]; // [rsp+80h] [rbp-430h] BYREF
  unsigned __int64 v65[6]; // [rsp+A0h] [rbp-410h] BYREF
  unsigned __int64 v66[4]; // [rsp+D0h] [rbp-3E0h] BYREF
  unsigned __int64 v67[6]; // [rsp+F0h] [rbp-3C0h] BYREF
  _QWORD v68[10]; // [rsp+120h] [rbp-390h] BYREF
  _BYTE v69[344]; // [rsp+170h] [rbp-340h] BYREF
  __int64 v70; // [rsp+2C8h] [rbp-1E8h]
  char *v71; // [rsp+2D0h] [rbp-1E0h] BYREF
  __int64 v72; // [rsp+2D8h] [rbp-1D8h]
  __int64 v73; // [rsp+2E0h] [rbp-1D0h]
  unsigned int v74; // [rsp+2E8h] [rbp-1C8h]
  char *v75; // [rsp+2F0h] [rbp-1C0h]
  int v76; // [rsp+2F8h] [rbp-1B8h]
  int v77; // [rsp+2FCh] [rbp-1B4h]
  char v78; // [rsp+300h] [rbp-1B0h] BYREF
  _BYTE v79[400]; // [rsp+320h] [rbp-190h] BYREF

  v5 = a3;
  v6 = HIDWORD(a3);
  sub_9BC330(
    (__int64)&v71,
    *(_QWORD *)(*(_QWORD *)(a1 + 416) + 32LL),
    (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 416) + 40LL) - *(_QWORD *)(*(_QWORD *)(a1 + 416) + 32LL)) >> 3,
    *(unsigned __int64 **)(a1 + 464),
    *(_QWORD *)(a1 + 448));
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
  v7 = v72;
  ++*(_QWORD *)(a1 + 16);
  ++v71;
  *(_QWORD *)(a1 + 24) = v7;
  v72 = 0;
  *(_QWORD *)(a1 + 32) = v73;
  v73 = 0;
  *(_DWORD *)(a1 + 40) = v74;
  v8 = v76;
  v74 = 0;
  if ( v76 )
  {
    v38 = *(_QWORD *)(a1 + 48);
    if ( v38 == a1 + 64 )
    {
      v10 = 0;
      v11 = 0;
    }
    else
    {
      _libc_free(v38);
      v8 = v76;
      v10 = v72;
      v11 = 16LL * v74;
    }
    *(_DWORD *)(a1 + 56) = v8;
    v39 = v77;
    *(_QWORD *)(a1 + 48) = v75;
    *(_DWORD *)(a1 + 60) = v39;
  }
  else
  {
    v9 = v75;
    *(_DWORD *)(a1 + 56) = 0;
    if ( v9 == &v78 )
    {
      v10 = 0;
      v11 = 0;
    }
    else
    {
      _libc_free((unsigned __int64)v9);
      v10 = v72;
      v11 = 16LL * v74;
    }
  }
  LODWORD(v12) = 0;
  sub_C7D6A0(v10, v11, 8);
  v57 = sub_2AB4370(a1);
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 16LL) + 216LL);
  if ( HIDWORD(v57) <= v13 )
  {
    _BitScanReverse64(&v14, v13 / HIDWORD(v57));
    v12 = 0x8000000000000000LL >> ((unsigned __int8)v14 ^ 0x3Fu);
  }
  v62 = sub_2AB9310((_QWORD *)a1, v12);
  if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 16LL) + 216LL) != 0xFFFFFFFFLL )
  {
    *(_DWORD *)(a1 + 116) = v12;
    *(_BYTE *)(a1 + 120) = 1;
  }
  if ( !v5 )
    goto LABEL_21;
  if ( !(_BYTE)v6 )
  {
    if ( (unsigned int)v12 >= v5 )
    {
      LODWORD(v71) = v5;
      BYTE4(v71) = 0;
      LODWORD(v72) = 0;
      BYTE4(v72) = 1;
    }
    else
    {
      v58 = *(__int64 **)(a1 + 480);
      if ( (unsigned __int8)sub_2AA9050(*v58) )
      {
        v40 = *(_QWORD *)(a1 + 416);
        v41 = **(_QWORD **)(v40 + 32);
        sub_D4BD20(&v60, v40, v15, v16, v17, v18);
        sub_B157E0((__int64)&v63, &v60);
        sub_B17850((__int64)&v71, (__int64)"loop-vectorize", (__int64)"VectorizationFactor", 19, &v63, v41);
        sub_B18290((__int64)&v71, "User-specified vectorization factor ", 0x24u);
        sub_B16C30((__int64)v66, "UserVectorizationFactor", 23, a3);
        v42 = sub_B826F0((__int64)&v71, (__int64)v66);
        sub_B18290(v42, " is unsafe, clamping to maximum safe vectorization factor ", 0x3Au);
        LODWORD(v61) = v12;
        BYTE4(v61) = 0;
        sub_B16C30((__int64)v64, "VectorizationFactor", 19, v61);
        v43 = sub_B826F0(v42, (__int64)v64);
        sub_23FE290((__int64)v68, v43, v44, v45, v46, v47);
        v70 = *(_QWORD *)(v43 + 424);
        v68[0] = &unk_49D9DE8;
        sub_2240A30(v65);
        sub_2240A30((unsigned __int64 *)v64);
        sub_2240A30(v67);
        sub_2240A30(v66);
        v71 = (char *)&unk_49D9D40;
        sub_23FD590((__int64)v79);
        sub_9C6650(&v60);
        sub_1049740(v58, (__int64)v68);
        v68[0] = &unk_49D9D40;
        sub_23FD590((__int64)v69);
      }
      LODWORD(v71) = v12;
      BYTE4(v71) = 0;
      LODWORD(v72) = 0;
      BYTE4(v72) = 1;
    }
    return v71;
  }
  if ( BYTE4(v62) && (unsigned int)v62 >= v5 )
  {
    LODWORD(v71) = v5;
    BYTE4(v71) = 0;
    LODWORD(v72) = v5;
    BYTE4(v72) = v6;
    return v71;
  }
  v20 = sub_DFE610(*(_QWORD *)(a1 + 448));
  v21 = *(__int64 **)(a1 + 480);
  v22 = *v21;
  if ( !v20 && !byte_500DD68 )
  {
    if ( !(unsigned __int8)sub_2AA9050(v22) )
      goto LABEL_21;
    v51 = *(_QWORD *)(a1 + 416);
    v54 = **(_QWORD **)(v51 + 32);
    sub_D4BD20(&v63, v51, v48, v49, v50, v54);
    sub_B157E0((__int64)v64, &v63);
    sub_B17850((__int64)&v71, (__int64)"loop-vectorize", (__int64)"VectorizationFactor", 19, v64, v54);
    sub_B18290((__int64)&v71, "User-specified vectorization factor ", 0x24u);
    sub_B16C30((__int64)v66, "UserVectorizationFactor", 23, a3);
    v52 = sub_B826F0((__int64)&v71, (__int64)v66);
    v28 = " is ignored because the target does not support scalable vectors. The compiler will pick a more suitable value.";
    v29 = 111;
    v30 = v52;
    goto LABEL_20;
  }
  if ( (unsigned __int8)sub_2AA9050(v22) )
  {
    v26 = *(_QWORD *)(a1 + 416);
    v53 = **(_QWORD **)(v26 + 32);
    sub_D4BD20(&v63, v26, v23, v24, v25, v53);
    sub_B157E0((__int64)v64, &v63);
    sub_B17850((__int64)&v71, (__int64)"loop-vectorize", (__int64)"VectorizationFactor", 19, v64, v53);
    sub_B18290((__int64)&v71, "User-specified vectorization factor ", 0x24u);
    sub_B16C30((__int64)v66, "UserVectorizationFactor", 23, a3);
    v27 = sub_B826F0((__int64)&v71, (__int64)v66);
    v28 = " is unsafe. Ignoring the hint to let the compiler pick a more suitable value.";
    v29 = 77;
    v30 = v27;
LABEL_20:
    sub_B18290(v30, v28, v29);
    sub_23FE290((__int64)v68, v30, v31, v32, v33, (__int64)v68);
    v70 = *(_QWORD *)(v30 + 424);
    v68[0] = &unk_49D9DE8;
    sub_2240A30(v67);
    sub_2240A30(v66);
    v71 = (char *)&unk_49D9D40;
    sub_23FD590((__int64)v79);
    sub_9C6650(&v63);
    sub_1049740(v21, (__int64)v68);
    v68[0] = &unk_49D9D40;
    sub_23FD590((__int64)v69);
  }
LABEL_21:
  LODWORD(v61) = v12;
  BYTE4(v61) = 0;
  v71 = (char *)sub_2AEBCD0((_QWORD *)a1, a2, v57, HIDWORD(v57), v61, a4);
  v34 = (int)v71;
  if ( (_DWORD)v71 )
  {
    v35 = BYTE4(v71);
  }
  else
  {
    v35 = 0;
    v34 = 1;
  }
  v36 = sub_2AEBCD0((_QWORD *)a1, a2, v57, HIDWORD(v57), v62, a4);
  v71 = (char *)v36;
  v37 = HIDWORD(v36);
  if ( (_DWORD)v36 )
  {
    if ( !BYTE4(v36) )
    {
      LOBYTE(v37) = 1;
      LODWORD(v36) = 0;
    }
  }
  else
  {
    LOBYTE(v37) = 1;
  }
  LODWORD(v71) = v34;
  BYTE4(v71) = v35;
  LODWORD(v72) = v36;
  BYTE4(v72) = v37;
  return v71;
}
