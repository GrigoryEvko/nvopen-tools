// Function: sub_2BD17B0
// Address: 0x2bd17b0
//
__int64 __fastcall sub_2BD17B0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rcx
  unsigned __int8 **v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int8 **v12; // rax
  unsigned __int8 **v13; // rcx
  int v14; // edx
  int v15; // edx
  int v16; // edx
  int v17; // edx
  unsigned int v18; // r12d
  int v20; // edx
  int v21; // edx
  int v22; // edx
  __int64 *v23; // r12
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // [rsp+8h] [rbp-4F8h]
  __int64 v33; // [rsp+8h] [rbp-4F8h]
  unsigned __int64 v34[2]; // [rsp+10h] [rbp-4F0h] BYREF
  _BYTE v35[48]; // [rsp+20h] [rbp-4E0h] BYREF
  __int64 *v36; // [rsp+50h] [rbp-4B0h] BYREF
  __int64 v37; // [rsp+58h] [rbp-4A8h]
  _BYTE v38[128]; // [rsp+60h] [rbp-4A0h] BYREF
  unsigned __int8 **v39; // [rsp+E0h] [rbp-420h] BYREF
  __int64 v40; // [rsp+E8h] [rbp-418h]
  _BYTE v41[128]; // [rsp+F0h] [rbp-410h] BYREF
  void *v42; // [rsp+170h] [rbp-390h] BYREF
  int v43; // [rsp+178h] [rbp-388h]
  char v44; // [rsp+17Ch] [rbp-384h]
  __int64 v45; // [rsp+180h] [rbp-380h]
  __m128i v46; // [rsp+188h] [rbp-378h]
  __int64 v47; // [rsp+198h] [rbp-368h]
  __m128i v48; // [rsp+1A0h] [rbp-360h]
  __m128i v49; // [rsp+1B0h] [rbp-350h]
  _QWORD v50[2]; // [rsp+1C0h] [rbp-340h] BYREF
  _BYTE v51[324]; // [rsp+1D0h] [rbp-330h] BYREF
  int v52; // [rsp+314h] [rbp-1ECh]
  __int64 v53; // [rsp+318h] [rbp-1E8h]
  void *v54; // [rsp+320h] [rbp-1E0h] BYREF
  int v55; // [rsp+328h] [rbp-1D8h]
  char v56; // [rsp+32Ch] [rbp-1D4h]
  __int64 v57; // [rsp+330h] [rbp-1D0h]
  __m128i v58; // [rsp+338h] [rbp-1C8h] BYREF
  __int64 v59; // [rsp+348h] [rbp-1B8h]
  __m128i v60; // [rsp+350h] [rbp-1B0h] BYREF
  __m128i v61; // [rsp+360h] [rbp-1A0h] BYREF
  _BYTE v62[8]; // [rsp+370h] [rbp-190h] BYREF
  int v63; // [rsp+378h] [rbp-188h]
  char v64; // [rsp+4C0h] [rbp-40h]
  int v65; // [rsp+4C4h] [rbp-3Ch]
  __int64 v66; // [rsp+4C8h] [rbp-38h]

  v37 = 0x1000000000LL;
  v40 = 0x1000000000LL;
  v32 = a6;
  v36 = (__int64 *)v38;
  v39 = (unsigned __int8 **)v41;
  v34[0] = (unsigned __int64)v35;
  v34[1] = 0xC00000000LL;
  if ( !(unsigned __int8)sub_2B3C490(a2, (__int64)&v39, (__int64)&v36, a5, a6, a7) )
    goto LABEL_33;
  v8 = 8LL * (unsigned int)v40;
  v9 = &v39[(unsigned __int64)v8 / 8];
  v10 = v8 >> 3;
  v11 = v8 >> 5;
  if ( !v11 )
  {
    v12 = v39;
LABEL_29:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          goto LABEL_32;
        goto LABEL_40;
      }
      v20 = **v12;
      if ( (_BYTE)v20 != 90 && (unsigned int)(v20 - 12) > 1 )
        goto LABEL_13;
      ++v12;
    }
    v21 = **v12;
    if ( (_BYTE)v21 != 90 && (unsigned int)(v21 - 12) > 1 )
      goto LABEL_13;
    ++v12;
LABEL_40:
    v22 = **v12;
    if ( (_BYTE)v22 != 90 && (unsigned int)(v22 - 12) > 1 )
      goto LABEL_13;
LABEL_32:
    v54 = (void *)sub_2B25EA0(v39, (unsigned int)v40, (__int64)v34);
    if ( BYTE4(v54) )
      goto LABEL_33;
    goto LABEL_14;
  }
  v12 = v39;
  v13 = &v39[4 * v11];
  while ( 1 )
  {
    v17 = **v12;
    if ( (_BYTE)v17 != 90 && (unsigned int)(v17 - 12) > 1 )
      break;
    v14 = *v12[1];
    if ( (_BYTE)v14 != 90 && (unsigned int)(v14 - 12) > 1 )
    {
      if ( v9 != v12 + 1 )
        goto LABEL_14;
      goto LABEL_32;
    }
    v15 = *v12[2];
    if ( (_BYTE)v15 != 90 && (unsigned int)(v15 - 12) > 1 )
    {
      v12 += 2;
      break;
    }
    v16 = *v12[3];
    if ( (_BYTE)v16 != 90 && (unsigned int)(v16 - 12) > 1 )
    {
      v12 += 3;
      break;
    }
    v12 += 4;
    if ( v12 == v13 )
    {
      v10 = v9 - v12;
      goto LABEL_29;
    }
  }
LABEL_13:
  if ( v9 == v12 )
    goto LABEL_32;
LABEL_14:
  if ( !v32 || (_DWORD)v37 != 2 )
  {
    v18 = sub_2BCE070(a1, v36, (unsigned int)v37, a5, v32, a3);
    goto LABEL_17;
  }
  v23 = *(__int64 **)(a5 + 3352);
  v33 = *v23;
  v24 = sub_B2BE50(*v23);
  if ( sub_B6EA50(v24)
    || (v30 = sub_B2BE50(v33),
        v31 = sub_B6F970(v30),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v31 + 48LL))(v31)) )
  {
    sub_B176B0((__int64)&v54, (__int64)"slp-vectorizer", (__int64)"NotPossible", 11, a2);
    sub_B18290(
      (__int64)&v54,
      "Cannot SLP vectorize list: only 2 elements of buildvector, trying reduction first.",
      0x52u);
    v27 = _mm_loadu_si128(&v58);
    v28 = _mm_loadu_si128(&v60);
    v43 = v55;
    v29 = _mm_loadu_si128(&v61);
    v46 = v27;
    v44 = v56;
    v48 = v28;
    v45 = v57;
    v42 = &unk_49D9D40;
    v49 = v29;
    v47 = v59;
    v50[0] = v51;
    v50[1] = 0x400000000LL;
    if ( v63 )
      sub_2B44C50((__int64)v50, (__int64)v62, v25, v26, (__int64)v50, (__int64)v62);
    v54 = &unk_49D9D40;
    v51[320] = v64;
    v52 = v65;
    v53 = v66;
    v42 = &unk_49D9DB0;
    sub_23FD590((__int64)v62);
    sub_1049740(v23, (__int64)&v42);
    v42 = &unk_49D9D40;
    sub_23FD590((__int64)v50);
  }
LABEL_33:
  v18 = 0;
LABEL_17:
  if ( (_BYTE *)v34[0] != v35 )
    _libc_free(v34[0]);
  if ( v39 != (unsigned __int8 **)v41 )
    _libc_free((unsigned __int64)v39);
  if ( v36 != (__int64 *)v38 )
    _libc_free((unsigned __int64)v36);
  return v18;
}
