// Function: sub_389EEE0
// Address: 0x389eee0
//
__int64 __fastcall sub_389EEE0(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        int a4,
        int a5,
        char a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15,
        char a16,
        char a17)
{
  __int64 v17; // r14
  __int64 v20; // rbx
  bool v21; // zf
  __int64 v22; // rdi
  unsigned int v23; // r8d
  const char *v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rdx
  const char *v30; // rax
  __int64 v31; // rdx
  unsigned __int8 v32; // r8
  void *v33; // rax
  unsigned __int64 v34; // rdi
  __int64 v35; // r12
  __int64 v36; // rsi
  __int64 v37; // rbx
  __int64 v38; // rdx
  unsigned int v39; // r11d
  __int64 v40; // r10
  __int64 v41; // rdi
  int *v42; // rax
  __int64 v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  _QWORD *v46; // r10
  _QWORD *v47; // r12
  __int16 v48; // dx
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // rax
  __int64 v52; // r14
  __int64 v53; // rbx
  __int64 v54; // rbx
  unsigned __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __m128i *v59; // rax
  unsigned int v60; // eax
  _BYTE *v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // rax
  unsigned __int64 v64; // [rsp+10h] [rbp-170h]
  __int16 *v65; // [rsp+18h] [rbp-168h]
  __int64 v66; // [rsp+18h] [rbp-168h]
  __int64 v67; // [rsp+20h] [rbp-160h]
  unsigned int v68; // [rsp+28h] [rbp-158h]
  __int64 v69; // [rsp+30h] [rbp-150h]
  _QWORD *v70; // [rsp+30h] [rbp-150h]
  unsigned int v71; // [rsp+30h] [rbp-150h]
  unsigned __int64 v72; // [rsp+38h] [rbp-148h]
  char v74; // [rsp+57h] [rbp-129h]
  unsigned __int8 v76; // [rsp+58h] [rbp-128h]
  unsigned __int8 v77; // [rsp+58h] [rbp-128h]
  unsigned __int8 v78; // [rsp+58h] [rbp-128h]
  unsigned __int8 v79; // [rsp+58h] [rbp-128h]
  unsigned __int8 v80; // [rsp+58h] [rbp-128h]
  unsigned __int8 v81; // [rsp+58h] [rbp-128h]
  unsigned __int8 v82; // [rsp+58h] [rbp-128h]
  __int64 v83; // [rsp+58h] [rbp-128h]
  unsigned __int8 v84; // [rsp+58h] [rbp-128h]
  __int64 v85; // [rsp+60h] [rbp-120h] BYREF
  __int64 v86; // [rsp+68h] [rbp-118h] BYREF
  __m128i **v87; // [rsp+70h] [rbp-110h] BYREF
  __int16 v88; // [rsp+80h] [rbp-100h]
  __int64 v89[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v90; // [rsp+A0h] [rbp-E0h] BYREF
  __m128i *v91; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v92; // [rsp+B8h] [rbp-C8h]
  __m128i v93; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD *v94; // [rsp+D0h] [rbp-B0h]
  __int64 v95; // [rsp+D8h] [rbp-A8h]
  _BYTE v96[16]; // [rsp+E0h] [rbp-A0h] BYREF
  _QWORD *v97; // [rsp+F0h] [rbp-90h]
  __int64 v98; // [rsp+F8h] [rbp-88h]
  _BYTE v99[16]; // [rsp+100h] [rbp-80h] BYREF
  unsigned __int64 v100; // [rsp+110h] [rbp-70h]
  unsigned int v101; // [rsp+118h] [rbp-68h]
  char v102; // [rsp+11Ch] [rbp-64h]
  void *v103; // [rsp+128h] [rbp-58h] BYREF
  __int64 v104; // [rsp+130h] [rbp-50h]
  __int64 v105; // [rsp+140h] [rbp-40h]
  unsigned __int64 v106; // [rsp+148h] [rbp-38h]

  v17 = a1 + 8;
  v20 = a1;
  v21 = *(_DWORD *)(a1 + 64) == 91;
  v22 = a1 + 8;
  if ( v21 )
  {
    *(_DWORD *)(v20 + 64) = sub_3887100(v22);
    if ( !a4 )
    {
      v74 = 1;
      goto LABEL_3;
    }
    v74 = 1;
    if ( (unsigned int)(a4 - 7) > 1 )
    {
      if ( (unsigned int)(a4 - 2) <= 3 )
        goto LABEL_3;
      v93.m128i_i8[1] = 1;
      v25 = "invalid linkage type for alias";
      goto LABEL_10;
    }
LABEL_8:
    if ( !a5 )
      goto LABEL_3;
    v93.m128i_i8[1] = 1;
    v25 = "symbol with local linkage must have default visibility";
LABEL_10:
    v91 = (__m128i *)v25;
    v93.m128i_i8[0] = 3;
    return (unsigned int)sub_38814C0(v17, a3, (__int64)&v91);
  }
  v74 = 0;
  *(_DWORD *)(v20 + 64) = sub_3887100(v22);
  if ( (unsigned int)(a4 - 7) <= 1 )
    goto LABEL_8;
LABEL_3:
  v72 = *(_QWORD *)(v20 + 56);
  v91 = (__m128i *)"expected type";
  v93.m128i_i16[0] = 259;
  if ( (unsigned __int8)sub_3891B00(v20, &v85, (__int64)&v91, 0)
    || (unsigned __int8)sub_388AF10(v20, 4, "expected comma after alias or ifunc's type") )
  {
    return 1;
  }
  v64 = *(_QWORD *)(v20 + 56);
  v26 = (unsigned int)(*(_DWORD *)(v20 + 64) - 264);
  if ( (unsigned int)v26 > 0x1D || (v31 = 536870925, !_bittest64(&v31, v26)) )
  {
    if ( !(unsigned __int8)sub_389C3E0((__int64 **)v20, &v86) )
      goto LABEL_14;
    return 1;
  }
  v99[0] = 0;
  v94 = v96;
  LODWORD(v91) = 0;
  v92 = 0;
  v93.m128i_i64[1] = 0;
  v95 = 0;
  v96[0] = 0;
  v97 = v99;
  v98 = 0;
  v101 = 1;
  v100 = 0;
  v102 = 0;
  a7 = 0;
  v65 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v89, 0.0);
  sub_169E320(&v103, v89, v65);
  sub_1698460((__int64)v89);
  v106 = 0;
  v32 = sub_389C540(v20, (__int64)&v91, 0.0, a8, a9);
  if ( v32 )
    goto LABEL_24;
  if ( (_DWORD)v91 == 11 )
  {
    v86 = v105;
    if ( v106 )
      j_j___libc_free_0_0(v106);
    if ( v103 == sub_16982C0() )
    {
      v51 = v104;
      if ( v104 )
      {
        if ( v104 != v104 + 32LL * *(_QWORD *)(v104 - 8) )
        {
          v67 = v17;
          v52 = v104;
          v66 = v20;
          v53 = v104 + 32LL * *(_QWORD *)(v104 - 8);
          do
          {
            v53 -= 32;
            sub_127D120((_QWORD *)(v53 + 8));
          }
          while ( v52 != v53 );
          v51 = v52;
          v20 = v66;
          v17 = v67;
        }
        j_j_j___libc_free_0_0(v51 - 8);
      }
    }
    else
    {
      sub_1698460((__int64)&v103);
    }
    if ( v101 > 0x40 && v100 )
      j_j___libc_free_0_0(v100);
    if ( v97 != (_QWORD *)v99 )
      j_j___libc_free_0((unsigned __int64)v97);
    if ( v94 != (_QWORD *)v96 )
      j_j___libc_free_0((unsigned __int64)v94);
LABEL_14:
    v27 = v86;
    v28 = *(_QWORD *)v86;
    if ( *(_BYTE *)(*(_QWORD *)v86 + 8LL) != 15 )
    {
      v91 = (__m128i *)"An alias or ifunc must have pointer type";
      v93.m128i_i16[0] = 259;
      return (unsigned int)sub_38814C0(v17, v64, (__int64)&v91);
    }
    v29 = *(_QWORD *)(v28 + 24);
    if ( v74 )
    {
      if ( v85 != v29 )
      {
        v93.m128i_i8[1] = 1;
        v30 = "explicit pointee type doesn't match operand's pointee type";
        goto LABEL_18;
      }
    }
    else if ( *(_BYTE *)(v29 + 8) != 12 )
    {
      v93.m128i_i8[1] = 1;
      v30 = "explicit pointee type should be a function type";
LABEL_18:
      v91 = (__m128i *)v30;
      v93.m128i_i8[0] = 3;
      return (unsigned int)sub_38814C0(v17, v72, (__int64)&v91);
    }
    v38 = a2[1];
    v39 = *(_DWORD *)(v28 + 8) >> 8;
    if ( v38 )
    {
      v71 = *(_DWORD *)(v28 + 8) >> 8;
      v58 = sub_1632000(*(_QWORD *)(v20 + 176), *a2, v38);
      v39 = v71;
      v40 = v58;
      if ( !v58 )
      {
        v27 = v86;
        goto LABEL_65;
      }
      v68 = v71;
      v69 = v58;
      if ( !sub_38942C0(v20 + 904, (__int64)a2) )
      {
        sub_8FD6D0((__int64)v89, "redefinition of global '@", a2);
        if ( v89[1] == 0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v59 = (__m128i *)sub_2241490((unsigned __int64 *)v89, "'", 1u);
        v91 = &v93;
        if ( (__m128i *)v59->m128i_i64[0] == &v59[1] )
        {
          v93 = _mm_loadu_si128(v59 + 1);
        }
        else
        {
          v91 = (__m128i *)v59->m128i_i64[0];
          v93.m128i_i64[0] = v59[1].m128i_i64[0];
        }
        v92 = v59->m128i_i64[1];
        v59->m128i_i64[0] = (__int64)v59[1].m128i_i64;
        v59->m128i_i64[1] = 0;
        v59[1].m128i_i8[0] = 0;
        v88 = 260;
        v87 = &v91;
        v60 = sub_38814C0(v17, a3, (__int64)&v87);
        v23 = v60;
        if ( v91 != &v93 )
        {
          v84 = v60;
          j_j___libc_free_0((unsigned __int64)v91);
          v23 = v84;
        }
        v34 = v89[0];
        if ( (__int64 *)v89[0] != &v90 )
          goto LABEL_34;
        return v23;
      }
    }
    else
    {
      v40 = *(_QWORD *)(v20 + 968);
      if ( !v40 )
        goto LABEL_65;
      v41 = v20 + 960;
      do
      {
        if ( (unsigned int)((__int64)(*(_QWORD *)(v20 + 1008) - *(_QWORD *)(v20 + 1000)) >> 3) > *(_DWORD *)(v40 + 32) )
        {
          v40 = *(_QWORD *)(v40 + 24);
        }
        else
        {
          v41 = v40;
          v40 = *(_QWORD *)(v40 + 16);
        }
      }
      while ( v40 );
      if ( v41 == v20 + 960
        || (unsigned int)((__int64)(*(_QWORD *)(v20 + 1008) - *(_QWORD *)(v20 + 1000)) >> 3) < *(_DWORD *)(v41 + 32) )
      {
        goto LABEL_65;
      }
      v68 = *(_DWORD *)(v28 + 8) >> 8;
      v69 = *(_QWORD *)(v41 + 40);
      v42 = sub_220F330((int *)v41, (_QWORD *)(v20 + 960));
      j_j___libc_free_0((unsigned __int64)v42);
      --*(_QWORD *)(v20 + 992);
    }
    v27 = v86;
    v40 = v69;
    v39 = v68;
LABEL_65:
    v93.m128i_i16[0] = 260;
    v70 = (_QWORD *)v40;
    v91 = (__m128i *)a2;
    if ( v74 )
      v43 = sub_15E57E0(v85, v39, a4, (__int64)&v91, v27, 0);
    else
      v43 = sub_15E5A30(v85, v39, a4, (__int64)&v91, v27, 0);
    v46 = v70;
    v47 = (_QWORD *)v43;
    v48 = *(_WORD *)(v43 + 32) & 0xE3CF | (16 * (a5 & 3)) | ((a16 & 7) << 10);
    *(_WORD *)(v43 + 32) = v48;
    if ( (v48 & 0xFu) - 7 <= 1 || (v48 & 0x30) != 0 && (v48 & 0xF) != 9 )
      *(_BYTE *)(v43 + 33) |= 0x40u;
    *(_WORD *)(v43 + 32) = *(_WORD *)(v43 + 32) & 0xFC3F | ((a17 & 3) << 6) | ((a6 & 3) << 8);
    if ( a15 )
      *(_BYTE *)(v43 + 33) |= 0x40u;
    if ( !a2[1] )
    {
      v91 = (__m128i *)v43;
      v61 = *(_BYTE **)(v20 + 1008);
      if ( v61 == *(_BYTE **)(v20 + 1016) )
      {
        sub_167C6C0(v20 + 1000, v61, &v91);
        v46 = v70;
      }
      else
      {
        if ( v61 )
        {
          *(_QWORD *)v61 = v43;
          v61 = *(_BYTE **)(v20 + 1008);
        }
        *(_QWORD *)(v20 + 1008) = v61 + 8;
      }
    }
    if ( v46 )
    {
      if ( *v47 != *v46 )
      {
        v91 = (__m128i *)"forward reference and definition of alias have different types";
        v93.m128i_i16[0] = 259;
        v82 = sub_38814C0(v17, v72, (__int64)&v91);
        sub_159D9E0((__int64)v47);
        sub_164BE60((__int64)v47, a7, a8, a9, a10, v49, v50, a13, a14);
        sub_1648B90((__int64)v47);
        return v82;
      }
      v83 = (__int64)v46;
      sub_164D160((__int64)v46, (__int64)v47, a7, a8, a9, a10, v44, v45, a13, a14);
      sub_15E5B20(v83);
    }
    v54 = *(_QWORD *)(v20 + 176);
    v55 = (unsigned __int64)(v47 + 6);
    if ( v74 )
    {
      sub_1631C60(v54 + 40, (__int64)v47);
      v56 = *(_QWORD *)(v54 + 40);
      v57 = v47[6];
      v23 = 0;
      v47[7] = v54 + 40;
      v56 &= 0xFFFFFFFFFFFFFFF8LL;
      v47[6] = v56 | v57 & 7;
      *(_QWORD *)(v56 + 8) = v55;
      *(_QWORD *)(v54 + 40) = v55 | *(_QWORD *)(v54 + 40) & 7LL;
    }
    else
    {
      sub_1631CE0(v54 + 56, (__int64)v47);
      v62 = *(_QWORD *)(v54 + 56);
      v63 = v47[6];
      v23 = 0;
      v47[7] = v54 + 56;
      v62 &= 0xFFFFFFFFFFFFFFF8LL;
      v47[6] = v62 | v63 & 7;
      *(_QWORD *)(v62 + 8) = v55;
      *(_QWORD *)(v54 + 56) = v55 | *(_QWORD *)(v54 + 56) & 7LL;
    }
    return v23;
  }
  v89[0] = (__int64)"invalid aliasee";
  LOWORD(v90) = 259;
  v32 = sub_38814C0(v17, v64, (__int64)v89);
LABEL_24:
  if ( v106 )
  {
    v76 = v32;
    j_j___libc_free_0_0(v106);
    v32 = v76;
  }
  v77 = v32;
  v33 = sub_16982C0();
  v23 = v77;
  if ( v103 == v33 )
  {
    v35 = v104;
    if ( v104 )
    {
      v36 = 32LL * *(_QWORD *)(v104 - 8);
      v37 = v104 + v36;
      if ( v104 != v104 + v36 )
      {
        do
        {
          v37 -= 32;
          sub_127D120((_QWORD *)(v37 + 8));
        }
        while ( v35 != v37 );
        LOBYTE(v23) = v77;
      }
      v81 = v23;
      j_j_j___libc_free_0_0(v35 - 8);
      v23 = v81;
    }
  }
  else
  {
    sub_1698460((__int64)&v103);
    v23 = v77;
  }
  if ( v101 > 0x40 && v100 )
  {
    v78 = v23;
    j_j___libc_free_0_0(v100);
    v23 = v78;
  }
  if ( v97 != (_QWORD *)v99 )
  {
    v79 = v23;
    j_j___libc_free_0((unsigned __int64)v97);
    v23 = v79;
  }
  v34 = (unsigned __int64)v94;
  if ( v94 != (_QWORD *)v96 )
  {
LABEL_34:
    v80 = v23;
    j_j___libc_free_0(v34);
    return v80;
  }
  return v23;
}
