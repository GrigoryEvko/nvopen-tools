// Function: sub_2E894A0
// Address: 0x2e894a0
//
__int64 __fastcall sub_2E894A0(__int64 a1, int a2)
{
  __m128i *v3; // rdi
  __m128i *v4; // rsi
  unsigned __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r9
  bool v8; // zf
  __m128i *v9; // rcx
  void (__fastcall *v10)(_BYTE *, _BYTE *, __int64); // rax
  void (__fastcall *v11)(_BYTE *, _BYTE *, __int64); // rax
  __m128i v12; // xmm0
  __m128i v13; // xmm2
  unsigned __int8 (__fastcall *v14)(__m128i *); // rcx
  __m128i v15; // xmm1
  __m128i *v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // r12
  __m128i *v21; // [rsp+8h] [rbp-268h]
  unsigned __int64 v22; // [rsp+18h] [rbp-258h]
  _DWORD v23[4]; // [rsp+20h] [rbp-250h] BYREF
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-240h]
  unsigned __int8 (__fastcall *v25)(__m128i *); // [rsp+38h] [rbp-238h]
  _QWORD v26[2]; // [rsp+40h] [rbp-230h] BYREF
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // [rsp+50h] [rbp-220h]
  unsigned __int8 (__fastcall *v28)(__m128i *); // [rsp+58h] [rbp-218h]
  _BYTE v29[16]; // [rsp+60h] [rbp-210h] BYREF
  void (__fastcall *v30)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-200h]
  unsigned __int8 (__fastcall *v31)(__m128i *); // [rsp+78h] [rbp-1F8h]
  _BYTE v32[16]; // [rsp+80h] [rbp-1F0h] BYREF
  void (__fastcall *v33)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-1E0h]
  unsigned __int8 (__fastcall *v34)(__m128i *); // [rsp+98h] [rbp-1D8h]
  _BYTE v35[16]; // [rsp+A0h] [rbp-1D0h] BYREF
  void (__fastcall *v36)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-1C0h]
  unsigned __int8 (__fastcall *v37)(__m128i *); // [rsp+B8h] [rbp-1B8h]
  _BYTE v38[16]; // [rsp+C0h] [rbp-1B0h] BYREF
  void (__fastcall *v39)(_BYTE *, _BYTE *, __int64); // [rsp+D0h] [rbp-1A0h]
  unsigned __int8 (__fastcall *v40)(__m128i *); // [rsp+D8h] [rbp-198h]
  __m128i *v41; // [rsp+E0h] [rbp-190h]
  __m128i *v42; // [rsp+E8h] [rbp-188h]
  __m128i v43; // [rsp+F0h] [rbp-180h] BYREF
  void (__fastcall *v44)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-170h]
  unsigned __int8 (__fastcall *v45)(__m128i *); // [rsp+108h] [rbp-168h]
  __m128i *v46; // [rsp+110h] [rbp-160h]
  __m128i *v47; // [rsp+118h] [rbp-158h]
  __m128i v48; // [rsp+120h] [rbp-150h] BYREF
  void (__fastcall *v49)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-140h]
  unsigned __int8 (__fastcall *v50)(__m128i *); // [rsp+138h] [rbp-138h]
  __m128i *v51; // [rsp+140h] [rbp-130h]
  __m128i *v52; // [rsp+148h] [rbp-128h]
  __m128i v53; // [rsp+150h] [rbp-120h] BYREF
  void (__fastcall *v54)(__m128i *, __m128i *, __int64); // [rsp+160h] [rbp-110h]
  unsigned __int8 (__fastcall *v55)(__m128i *, __m128i *); // [rsp+168h] [rbp-108h]
  __m128i *v56; // [rsp+170h] [rbp-100h]
  __m128i *v57; // [rsp+178h] [rbp-F8h]
  __m128i v58; // [rsp+180h] [rbp-F0h] BYREF
  void (__fastcall *v59)(__m128i *, __m128i *, __int64); // [rsp+190h] [rbp-E0h]
  unsigned __int8 (__fastcall *v60)(__m128i *); // [rsp+198h] [rbp-D8h]
  _BYTE *v61; // [rsp+1A0h] [rbp-D0h] BYREF
  __int64 v62; // [rsp+1A8h] [rbp-C8h]
  _BYTE v63[48]; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i *v64; // [rsp+1E0h] [rbp-90h]
  __m128i *v65; // [rsp+1E8h] [rbp-88h]
  __m128i v66; // [rsp+1F0h] [rbp-80h] BYREF
  void (__fastcall *v67)(__m128i *, __m128i *, __int64); // [rsp+200h] [rbp-70h]
  unsigned __int8 (__fastcall *v68)(__m128i *); // [rsp+208h] [rbp-68h]
  __m128i *v69; // [rsp+210h] [rbp-60h]
  __m128i *v70; // [rsp+218h] [rbp-58h]
  __m128i v71; // [rsp+220h] [rbp-50h] BYREF
  void (__fastcall *v72)(__m128i *, __m128i *, __int64); // [rsp+230h] [rbp-40h]
  unsigned __int8 (__fastcall *v73)(__m128i *); // [rsp+238h] [rbp-38h]

  v3 = (__m128i *)v26;
  v61 = v63;
  v62 = 0x600000000LL;
  v23[0] = a2;
  v4 = (__m128i *)v23;
  v25 = (unsigned __int8 (__fastcall *)(__m128i *))sub_2E85480;
  v27 = 0;
  v24 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))sub_2E854A0;
  sub_2E854A0(v26, v23, 2);
  v8 = *(_WORD *)(a1 + 68) == 14;
  v28 = v25;
  v27 = v24;
  v9 = *(__m128i **)(a1 + 32);
  if ( v8 )
  {
    v21 = *(__m128i **)(a1 + 32);
    v22 = (unsigned __int64)&v9[2].m128i_u64[1];
  }
  else
  {
    v5 = (unsigned __int64)v9 + 40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
    v22 = v5;
    v21 = v9 + 5;
  }
  v30 = 0;
  if ( v24 && (v4 = (__m128i *)v26, v3 = (__m128i *)v29, v24(v29, v26, 2), v39 = 0, v31 = v28, (v30 = v27) != 0) )
  {
    v4 = (__m128i *)v29;
    v3 = (__m128i *)v38;
    v27(v38, v29, 2);
    v44 = 0;
    v40 = v31;
    v41 = (__m128i *)v22;
    v39 = v30;
    v42 = (__m128i *)v22;
    if ( v30 )
    {
      v3 = &v43;
      v30(&v43, v38, 2);
      v4 = v41;
      v45 = v40;
      v10 = v39;
      v44 = v39;
      if ( v42 != v41 )
      {
        while ( 1 )
        {
          if ( !v10 )
            goto LABEL_64;
          v3 = &v43;
          if ( v45(&v43) )
            break;
          v4 = (__m128i *)((char *)v41 + 40);
          v41 = v4;
          if ( v42 == v4 )
            break;
          v10 = v44;
        }
        v10 = v39;
      }
      if ( v10 )
      {
        v4 = (__m128i *)v38;
        v3 = (__m128i *)v38;
        v10(v38, v38, 3);
      }
    }
  }
  else
  {
    v44 = 0;
    v41 = (__m128i *)v22;
    v42 = (__m128i *)v22;
  }
  v33 = 0;
  if ( !v27 )
  {
    v36 = 0;
    goto LABEL_62;
  }
  v4 = (__m128i *)v26;
  v3 = (__m128i *)v32;
  v27(v32, v26, 2);
  v36 = 0;
  v34 = v28;
  v33 = v27;
  if ( !v27 )
  {
LABEL_62:
    v49 = 0;
    v46 = v21;
    v47 = (__m128i *)v22;
    goto LABEL_63;
  }
  v4 = (__m128i *)v32;
  v3 = (__m128i *)v35;
  v27(v35, v32, 2);
  v49 = 0;
  v37 = v34;
  v46 = v21;
  v36 = v33;
  v47 = (__m128i *)v22;
  if ( v33 )
  {
    v3 = &v48;
    v33(&v48, v35, 2);
    v4 = v46;
    v50 = v37;
    v11 = v36;
    v49 = v36;
    if ( v46 != v47 )
    {
      while ( 1 )
      {
        if ( !v11 )
          goto LABEL_64;
        v3 = &v48;
        if ( v50(&v48) )
          break;
        v4 = (__m128i *)((char *)v46 + 40);
        v46 = v4;
        if ( v47 == v4 )
          break;
        v11 = v49;
      }
      v11 = v36;
    }
    if ( v11 )
    {
      v3 = (__m128i *)v35;
      v11(v35, v35, 3);
    }
    goto LABEL_27;
  }
LABEL_63:
  if ( (__m128i *)v22 != v21 )
    goto LABEL_64;
LABEL_27:
  v12 = _mm_loadu_si128(&v43);
  v13 = _mm_loadu_si128(&v58);
  v51 = v46;
  v4 = (__m128i *)v44;
  v64 = v46;
  v56 = v41;
  v14 = v45;
  v65 = v47;
  v69 = v41;
  v15 = _mm_loadu_si128(&v48);
  v67 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v49;
  v70 = v42;
  v72 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v44;
  v44 = 0;
  v45 = 0;
  v68 = v50;
  v73 = v14;
  v43 = v13;
  v58 = v12;
  v53 = v15;
  v66 = v15;
  v71 = v12;
  if ( v33 )
  {
    v3 = (__m128i *)v32;
    v4 = (__m128i *)v32;
    v33(v32, v32, 3);
    if ( v44 )
    {
      v3 = &v43;
      v4 = &v43;
      v44(&v43, &v43, 3);
    }
  }
  if ( v30 )
  {
    v3 = (__m128i *)v29;
    v4 = (__m128i *)v29;
    v30(v29, v29, 3);
  }
  if ( v27 )
  {
    v4 = (__m128i *)v26;
    v3 = (__m128i *)v26;
    v27(v26, v26, 3);
  }
  if ( v24 )
  {
    v4 = (__m128i *)v23;
    v3 = (__m128i *)v23;
    v24(v23, v23, 3);
  }
  v54 = 0;
  v51 = v64;
  v52 = v65;
  if ( v67 )
  {
    v4 = &v66;
    v3 = &v53;
    v67(&v53, &v66, 2);
    v55 = (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))v68;
    v54 = v67;
  }
  v59 = 0;
  v56 = v69;
  v57 = v70;
  if ( v72 )
  {
    v4 = &v71;
    v3 = &v58;
    v72(&v58, &v71, 2);
    v60 = v73;
    v59 = v72;
  }
LABEL_40:
  v16 = v51;
  if ( v51 != v56 )
  {
    while ( 1 )
    {
      v17 = (unsigned int)v62;
      v18 = (unsigned int)v62 + 1LL;
      if ( v18 > HIDWORD(v62) )
      {
        v4 = (__m128i *)v63;
        v3 = (__m128i *)&v61;
        sub_C8D5F0((__int64)&v61, v63, v18, 8u, v6, v7);
        v17 = (unsigned int)v62;
      }
      v5 = (unsigned __int64)v61;
      *(_QWORD *)&v61[8 * v17] = v16;
      LODWORD(v62) = v62 + 1;
      v16 = (__m128i *)((char *)v51 + 40);
      v51 = v16;
      if ( v16 != v52 )
        break;
LABEL_47:
      if ( v56 == v16 )
        goto LABEL_48;
    }
    while ( v54 )
    {
      v4 = v16;
      v3 = &v53;
      if ( v55(&v53, v16) )
        goto LABEL_40;
      v16 = (__m128i *)((char *)v51 + 40);
      v51 = v16;
      if ( v52 == v16 )
        goto LABEL_47;
    }
LABEL_64:
    sub_4263D6(v3, v4, v5);
  }
LABEL_48:
  if ( v59 )
    v59(&v58, &v58, 3);
  if ( v54 )
    v54(&v53, &v53, 3);
  if ( v72 )
    v72(&v71, &v71, 3);
  if ( v67 )
    v67(&v66, &v66, 3);
  v19 = sub_2E893B0(a1, (__int64)&v61);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  return v19;
}
