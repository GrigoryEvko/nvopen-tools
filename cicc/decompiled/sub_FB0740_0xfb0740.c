// Function: sub_FB0740
// Address: 0xfb0740
//
__int64 __fastcall sub_FB0740(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, _BYTE *a6, __int64 a7)
{
  unsigned __int8 *v7; // r12
  __int64 v10; // rax
  __int64 *v11; // rdi
  __m128i *v12; // rax
  __int64 v13; // rdx
  int v14; // esi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __int64 *v23; // r14
  __int64 *v24; // r10
  int v25; // r15d
  unsigned __int8 *v26; // rbx
  __int64 v27; // rdx
  unsigned __int8 *v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 *v38; // rax
  unsigned int v40; // esi
  unsigned int v41; // eax
  unsigned int v42; // ecx
  unsigned int v43; // r8d
  __int64 v44; // rax
  _BYTE *v45; // r14
  bool v46; // al
  __int64 *v47; // r10
  _BYTE *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  unsigned int v51; // eax
  __int64 v52; // r9
  __int64 v53; // r15
  __int64 *v54; // r12
  __int64 v55; // r13
  _BYTE *v56; // r14
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  unsigned int v59; // eax
  __int64 *v60; // rsi
  __int64 *v61; // r10
  __int64 v62; // rdx
  unsigned __int8 *v63; // [rsp+0h] [rbp-210h]
  __int64 v64; // [rsp+8h] [rbp-208h]
  __int64 *v65; // [rsp+10h] [rbp-200h]
  __int64 *v66; // [rsp+18h] [rbp-1F8h]
  _BYTE *v67; // [rsp+18h] [rbp-1F8h]
  __int64 *v68; // [rsp+18h] [rbp-1F8h]
  __int64 v72; // [rsp+40h] [rbp-1D0h]
  __int64 v73; // [rsp+48h] [rbp-1C8h]
  unsigned __int8 *v74; // [rsp+50h] [rbp-1C0h] BYREF
  _BYTE *v75; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 *v76; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v77; // [rsp+68h] [rbp-1A8h]
  _BYTE v78[32]; // [rsp+70h] [rbp-1A0h] BYREF
  __m128i v79; // [rsp+90h] [rbp-180h]
  __m128i v80; // [rsp+A0h] [rbp-170h]
  _BYTE v81[16]; // [rsp+B0h] [rbp-160h] BYREF
  void (__fastcall *v82)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-150h]
  unsigned __int8 (__fastcall *v83)(unsigned __int8 *, __int64 *); // [rsp+C8h] [rbp-148h]
  _OWORD v84[2]; // [rsp+D0h] [rbp-140h] BYREF
  _BYTE v85[16]; // [rsp+F0h] [rbp-120h] BYREF
  void (__fastcall *v86)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-110h]
  __int64 v87; // [rsp+108h] [rbp-108h]
  __int64 v88; // [rsp+110h] [rbp-100h] BYREF
  __int64 v89; // [rsp+118h] [rbp-F8h]
  __int64 *v90; // [rsp+120h] [rbp-F0h] BYREF
  unsigned int v91; // [rsp+128h] [rbp-E8h]
  __m128i v92; // [rsp+160h] [rbp-B0h] BYREF
  __m128i v93; // [rsp+170h] [rbp-A0h] BYREF
  _BYTE v94[16]; // [rsp+180h] [rbp-90h] BYREF
  void (__fastcall *v95)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-80h]
  unsigned __int8 (__fastcall *v96)(_QWORD, _QWORD); // [rsp+198h] [rbp-78h]
  __m128i v97; // [rsp+1A0h] [rbp-70h] BYREF
  __m128i v98; // [rsp+1B0h] [rbp-60h] BYREF
  _BYTE v99[16]; // [rsp+1C0h] [rbp-50h] BYREF
  void (__fastcall *v100)(_BYTE *, _BYTE *, __int64); // [rsp+1D0h] [rbp-40h]
  __int64 v101; // [rsp+1D8h] [rbp-38h]

  v10 = *(_QWORD *)(a1 + 40);
  v11 = (__int64 *)&v90;
  v88 = 0;
  v89 = 1;
  v73 = v10;
  v12 = (__m128i *)&v90;
  do
  {
    v12->m128i_i64[0] = -4096;
    ++v12;
  }
  while ( v12 != &v92 );
  v13 = **(_QWORD **)(a1 - 8);
  v92.m128i_i64[1] = a2;
  v92.m128i_i64[0] = v13;
  if ( (v89 & 1) != 0 )
  {
    v14 = 3;
  }
  else
  {
    v40 = v91;
    v11 = v90;
    if ( !v91 )
    {
      v41 = v89;
      ++v88;
      *(_QWORD *)&v84[0] = 0;
      v42 = ((unsigned int)v89 >> 1) + 1;
LABEL_59:
      v43 = 3 * v40;
      goto LABEL_60;
    }
    v14 = v91 - 1;
  }
  v15 = v14 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v16 = &v11[2 * v15];
  v17 = *v16;
  if ( v13 == *v16 )
    goto LABEL_6;
  LODWORD(v7) = 1;
  v61 = 0;
  while ( v17 != -4096 )
  {
    if ( v17 != -8192 || v61 )
      v16 = v61;
    v15 = v14 & ((_DWORD)v7 + v15);
    v17 = v11[2 * v15];
    if ( v13 == v17 )
      goto LABEL_6;
    LODWORD(v7) = (_DWORD)v7 + 1;
    v61 = v16;
    v16 = &v11[2 * v15];
  }
  if ( !v61 )
    v61 = v16;
  v41 = v89;
  ++v88;
  *(_QWORD *)&v84[0] = v61;
  v42 = ((unsigned int)v89 >> 1) + 1;
  if ( (v89 & 1) == 0 )
  {
    v40 = v91;
    goto LABEL_59;
  }
  v43 = 12;
  v40 = 4;
LABEL_60:
  if ( 4 * v42 >= v43 )
  {
    v40 *= 2;
    goto LABEL_99;
  }
  if ( v40 - HIDWORD(v89) - v42 <= v40 >> 3 )
  {
LABEL_99:
    v7 = (unsigned __int8 *)&v88;
    sub_FAFFE0((__int64)&v88, v40);
    sub_F9ED60((__int64)&v88, v92.m128i_i64, v84);
    v13 = v92.m128i_i64[0];
    v41 = v89;
  }
  LODWORD(v89) = (2 * (v41 >> 1) + 2) | v41 & 1;
  v44 = *(_QWORD *)&v84[0];
  if ( **(_QWORD **)&v84[0] != -4096 )
    --HIDWORD(v89);
  **(_QWORD **)&v84[0] = v13;
  *(_QWORD *)(v44 + 8) = v92.m128i_i64[1];
LABEL_6:
  v18 = (__int64 *)a3;
  sub_AA72C0(&v92, a3, 0);
  v82 = 0;
  v79 = _mm_load_si128(&v92);
  v80 = _mm_load_si128(&v93);
  if ( v95 )
  {
    v18 = (__int64 *)v94;
    v95(v81, v94, 2);
    v83 = v96;
    v82 = v95;
  }
  v21 = _mm_load_si128(&v97);
  v22 = _mm_load_si128(&v98);
  v86 = 0;
  v84[0] = v21;
  v84[1] = v22;
  if ( v100 )
  {
    v18 = (__int64 *)v99;
    v100(v85, v99, 2);
    v87 = v101;
    v86 = v100;
  }
  v23 = (__int64 *)v79.m128i_i64[0];
  v24 = (__int64 *)v79.m128i_i64[0];
  if ( *(_QWORD *)&v84[0] == v79.m128i_i64[0] )
    goto LABEL_27;
  v7 = v81;
  while ( 2 )
  {
    if ( !v24 )
      BUG();
    v25 = *((unsigned __int8 *)v24 - 24);
    v26 = (unsigned __int8 *)(v24 - 3);
    v27 = (unsigned int)(v25 - 30);
    if ( (unsigned int)v27 <= 0xA )
    {
      if ( (unsigned int)sub_B46E30((__int64)(v24 - 3)) != 1 )
        goto LABEL_44;
      if ( (unsigned int)(v25 - 29) > 6 )
      {
        if ( (unsigned int)(v25 - 37) <= 3 )
          goto LABEL_44;
      }
      else if ( (unsigned int)(v25 - 29) > 4 )
      {
        goto LABEL_44;
      }
      v18 = 0;
      v28 = v26;
      v73 = a3;
      a3 = sub_B46EC0((__int64)v26, 0);
      goto LABEL_18;
    }
    if ( (_BYTE)v25 == 86 )
    {
      v45 = (_BYTE *)*(v24 - 15);
      if ( *v45 > 0x15u )
      {
        v18 = &v88;
        v45 = (_BYTE *)sub_F8E490(*(v24 - 15), (__int64)&v88);
        if ( !v45 )
          break;
      }
      v66 = v24;
      v46 = sub_AD7930(v45, (__int64)v18, v27, v19, v20);
      v47 = v66;
      if ( v46 )
      {
        v48 = (_BYTE *)*(v66 - 11);
        if ( *v48 > 0x15u )
          goto LABEL_106;
      }
      else
      {
        if ( !sub_AC30F0((__int64)v45) )
          break;
        v47 = v66;
        v48 = (_BYTE *)*(v66 - 7);
        if ( *v48 > 0x15u )
        {
LABEL_106:
          v48 = (_BYTE *)sub_F8E490((__int64)v48, (__int64)&v88);
          goto LABEL_90;
        }
      }
      goto LABEL_71;
    }
    v76 = (__int64 *)v78;
    v77 = 0x400000000LL;
    v51 = *((_DWORD *)v24 - 5) & 0x7FFFFFF;
    if ( !v51 )
    {
      v60 = (__int64 *)v78;
      v62 = 0;
      goto LABEL_101;
    }
    v64 = a3;
    v52 = 32LL * v51;
    v63 = v7;
    v53 = 0;
    v54 = v24;
    v55 = v52;
    do
    {
      if ( (*((_BYTE *)v54 - 17) & 0x40) != 0 )
      {
        v56 = *(_BYTE **)(*(v54 - 4) + v53);
        if ( *v56 <= 0x15u )
          goto LABEL_81;
      }
      else
      {
        v56 = *(_BYTE **)&v26[v53 + -32 * (*((_DWORD *)v54 - 5) & 0x7FFFFFF)];
        if ( *v56 <= 0x15u )
          goto LABEL_81;
      }
      v60 = &v88;
      v48 = (_BYTE *)sub_F8E490((__int64)v56, (__int64)&v88);
      v56 = v48;
      if ( !v48 )
      {
        v47 = v54;
        a3 = v64;
        v7 = v63;
        goto LABEL_88;
      }
LABEL_81:
      v57 = (unsigned int)v77;
      v58 = (unsigned int)v77 + 1LL;
      if ( v58 > HIDWORD(v77) )
      {
        sub_C8D5F0((__int64)&v76, v78, v58, 8u, v20, v52);
        v57 = (unsigned int)v77;
      }
      v53 += 32;
      v76[v57] = (__int64)v56;
      v59 = v77 + 1;
      LODWORD(v77) = v77 + 1;
    }
    while ( v55 != v53 );
    v24 = v54;
    a3 = v64;
    v7 = v63;
    v62 = v59;
    v60 = v76;
LABEL_101:
    v68 = v24;
    v48 = (_BYTE *)sub_97D230(v26, v60, v62, a6, 0, 1u);
    v47 = v68;
LABEL_88:
    if ( v76 != (__int64 *)v78 )
    {
      v65 = v47;
      v67 = v48;
      _libc_free(v76, v60);
      v47 = v65;
      v48 = v67;
    }
LABEL_90:
    if ( v48 )
    {
LABEL_71:
      v49 = *(v47 - 1);
      if ( v49 )
      {
        while ( 1 )
        {
          v50 = *(_QWORD *)(v49 + 24);
          if ( *(_BYTE *)v50 <= 0x1Cu
            || *(_QWORD *)(v50 + 40) != a3
            && (*(_BYTE *)v50 != 84
             || *(_QWORD *)(*(_QWORD *)(v50 - 8)
                          + 32LL * *(unsigned int *)(v50 + 72)
                          + 8LL * (unsigned int)((v49 - *(_QWORD *)(v50 - 8)) >> 5)) != a3) )
          {
            break;
          }
          v49 = *(_QWORD *)(v49 + 8);
          if ( !v49 )
            goto LABEL_77;
        }
LABEL_44:
        if ( v86 )
          v86(v85, v85, 3);
        if ( v82 )
          v82(v81, v81, 3);
        if ( v100 )
          v100(v99, v99, 3);
        if ( v95 )
          v95(v94, v94, 3);
        goto LABEL_52;
      }
LABEL_77:
      v28 = (unsigned __int8 *)&v76;
      v74 = v26;
      v18 = &v88;
      v75 = v48;
      sub_FB0400((__int64)&v76, (__int64)&v88, (__int64 *)&v74, (__int64 *)&v75);
      v23 = (__int64 *)v79.m128i_i64[0];
LABEL_18:
      v23 = (__int64 *)v23[1];
      v19 = 0;
      v79.m128i_i16[4] = 0;
      v79.m128i_i64[0] = (__int64)v23;
      v24 = v23;
      if ( v23 == (__int64 *)v80.m128i_i64[0] )
      {
LABEL_26:
        if ( *(__int64 **)&v84[0] == v24 )
          break;
      }
      else
      {
        v18 = v23;
        do
        {
          if ( v18 )
            v18 -= 3;
          if ( !v82 )
            sub_4263D6(v28, v18, v29);
          v28 = v7;
          if ( v83(v7, v18) )
          {
            v24 = (__int64 *)v79.m128i_i64[0];
            v23 = (__int64 *)v79.m128i_i64[0];
            goto LABEL_26;
          }
          v29 = 0;
          v18 = *(__int64 **)(v79.m128i_i64[0] + 8);
          v79.m128i_i16[4] = 0;
          v79.m128i_i64[0] = (__int64)v18;
          v23 = v18;
        }
        while ( (__int64 *)v80.m128i_i64[0] != v18 );
        v24 = v18;
        if ( *(__int64 **)&v84[0] == v18 )
          break;
      }
      continue;
    }
    break;
  }
LABEL_27:
  sub_A17130((__int64)v85);
  sub_A17130((__int64)v81);
  sub_A17130((__int64)v99);
  sub_A17130((__int64)v94);
  if ( *a4 )
  {
    if ( *a4 == a3 )
      goto LABEL_29;
LABEL_52:
    LODWORD(v7) = 0;
  }
  else
  {
    *a4 = a3;
LABEL_29:
    v92.m128i_i64[0] = sub_AA5930(a3);
    if ( v92.m128i_i64[0] != v30 )
    {
      v72 = v30;
      v31 = v92.m128i_i64[0];
      do
      {
        if ( (*(_DWORD *)(v31 + 4) & 0x7FFFFFF) != 0 )
        {
          v32 = *(_QWORD *)(v31 - 8);
          v33 = 0;
          while ( v73 != *(_QWORD *)(v32 + 32LL * *(unsigned int *)(v31 + 72) + 8 * v33) )
          {
            if ( (*(_DWORD *)(v31 + 4) & 0x7FFFFFF) == (_DWORD)++v33 )
              goto LABEL_41;
          }
          v34 = 32 * v33;
          v7 = *(unsigned __int8 **)(v32 + v34);
          if ( !v7 )
            BUG();
          if ( *v7 > 0x15u )
          {
            v7 = (unsigned __int8 *)sub_F8E490(*(_QWORD *)(v32 + v34), (__int64)&v88);
            if ( !v7 )
              goto LABEL_52;
          }
          if ( !(unsigned __int8)sub_F8ECA0(v7, a7) )
            goto LABEL_52;
          v37 = *(unsigned int *)(a5 + 8);
          if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
          {
            sub_C8D5F0(a5, (const void *)(a5 + 16), v37 + 1, 0x10u, v35, v36);
            v37 = *(unsigned int *)(a5 + 8);
          }
          v38 = (__int64 *)(*(_QWORD *)a5 + 16 * v37);
          *v38 = v31;
          v38[1] = (__int64)v7;
          ++*(_DWORD *)(a5 + 8);
        }
LABEL_41:
        sub_F8F2F0((__int64)&v92);
        v31 = v92.m128i_i64[0];
      }
      while ( v72 != v92.m128i_i64[0] );
    }
    LOBYTE(v7) = *(_DWORD *)(a5 + 8) != 0;
  }
  if ( (v89 & 1) == 0 )
    sub_C7D6A0((__int64)v90, 16LL * v91, 8);
  return (unsigned int)v7;
}
