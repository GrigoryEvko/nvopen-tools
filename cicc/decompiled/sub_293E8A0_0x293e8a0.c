// Function: sub_293E8A0
// Address: 0x293e8a0
//
__int64 __fastcall sub_293E8A0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // r14d
  __int64 v6; // r9
  __int64 v7; // rbx
  _QWORD *v8; // rax
  _BYTE *v9; // rdx
  unsigned int v10; // r8d
  _QWORD *i; // rdx
  __m128i v12; // rax
  __m128i v13; // rax
  unsigned int v14; // esi
  unsigned int j; // r14d
  __m128i *v16; // rdx
  char v17; // al
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __m128i v22; // rax
  __int64 v23; // rax
  __int64 v24; // r9
  int v25; // r8d
  __m128i *v26; // rax
  __m128i *v27; // rdx
  __m128i *k; // rdx
  unsigned int v29; // ebx
  unsigned __int32 m; // r14d
  unsigned int n; // r15d
  unsigned int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rdx
  __m128i v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  char v39; // al
  __m128i *v40; // rcx
  __m128i v41; // rax
  char v42; // al
  __m128i *v43; // rcx
  __int64 **v44; // rcx
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  __m128i v47; // xmm1
  unsigned int v48; // r14d
  unsigned int v49; // r15d
  __int64 ii; // r10
  __int64 *v51; // r10
  __m128i v52; // rax
  unsigned __int64 v53; // rax
  unsigned __int32 v54; // r12d
  unsigned int v55; // esi
  __int64 v56; // r8
  __int64 v57; // rax
  __m128i v58; // xmm5
  __m128i v59; // xmm3
  unsigned __int8 v60; // [rsp+17h] [rbp-3F9h]
  __int64 v61; // [rsp+20h] [rbp-3F0h]
  __int64 v62; // [rsp+28h] [rbp-3E8h]
  unsigned __int8 v63; // [rsp+28h] [rbp-3E8h]
  unsigned int v64; // [rsp+30h] [rbp-3E0h]
  unsigned __int8 v65; // [rsp+30h] [rbp-3E0h]
  unsigned int v66; // [rsp+40h] [rbp-3D0h]
  __int64 v67; // [rsp+58h] [rbp-3B8h]
  _BYTE *v68; // [rsp+58h] [rbp-3B8h]
  bool v69; // [rsp+60h] [rbp-3B0h]
  unsigned __int64 v70; // [rsp+60h] [rbp-3B0h]
  __int64 **v71; // [rsp+68h] [rbp-3A8h]
  int v72; // [rsp+78h] [rbp-398h]
  __int64 v73; // [rsp+80h] [rbp-390h] BYREF
  __int32 v74; // [rsp+88h] [rbp-388h]
  unsigned int v75; // [rsp+8Ch] [rbp-384h]
  __int64 v76; // [rsp+90h] [rbp-380h]
  __int64 v77; // [rsp+98h] [rbp-378h]
  __int64 v78; // [rsp+A0h] [rbp-370h] BYREF
  __int32 v79; // [rsp+A8h] [rbp-368h]
  unsigned int v80; // [rsp+ACh] [rbp-364h]
  __int64 **v81; // [rsp+B0h] [rbp-360h]
  __int64 **v82; // [rsp+B8h] [rbp-358h]
  char v83; // [rsp+C0h] [rbp-350h]
  __m128i v84; // [rsp+D0h] [rbp-340h] BYREF
  __int64 v85; // [rsp+E0h] [rbp-330h]
  __int64 v86; // [rsp+E8h] [rbp-328h]
  unsigned __int8 v87; // [rsp+F0h] [rbp-320h]
  __m128i v88; // [rsp+100h] [rbp-310h] BYREF
  __m128i v89; // [rsp+110h] [rbp-300h] BYREF
  __int64 v90; // [rsp+120h] [rbp-2F0h]
  __m128i v91; // [rsp+130h] [rbp-2E0h] BYREF
  __m128i v92; // [rsp+140h] [rbp-2D0h] BYREF
  __int64 v93; // [rsp+150h] [rbp-2C0h]
  __m128i v94; // [rsp+160h] [rbp-2B0h] BYREF
  __m128i v95; // [rsp+170h] [rbp-2A0h] BYREF
  __int64 v96; // [rsp+180h] [rbp-290h]
  __m128i v97; // [rsp+190h] [rbp-280h] BYREF
  __m128i v98; // [rsp+1A0h] [rbp-270h]
  __int64 v99; // [rsp+1B0h] [rbp-260h]
  _BYTE *v100; // [rsp+1C0h] [rbp-250h] BYREF
  __int64 v101; // [rsp+1C8h] [rbp-248h]
  _BYTE v102[64]; // [rsp+1D0h] [rbp-240h] BYREF
  __int64 v103[2]; // [rsp+210h] [rbp-200h] BYREF
  char v104; // [rsp+220h] [rbp-1F0h] BYREF
  void *v105; // [rsp+290h] [rbp-180h]
  __m128i v106[5]; // [rsp+2A0h] [rbp-170h] BYREF
  char *v107; // [rsp+2F0h] [rbp-120h]
  char v108; // [rsp+300h] [rbp-110h] BYREF
  __m128i v109; // [rsp+340h] [rbp-D0h] BYREF
  __m128i v110; // [rsp+350h] [rbp-C0h] BYREF
  __int64 v111; // [rsp+360h] [rbp-B0h]
  char *v112; // [rsp+390h] [rbp-80h]
  char v113; // [rsp+3A0h] [rbp-70h] BYREF

  v3 = a1;
  if ( *(_DWORD *)(a1 + 1152) && !sub_293A020(a1, (unsigned __int8 *)a2) )
    return 0;
  sub_2939E80((__int64)&v78, a1, *(_QWORD *)(a2 + 8));
  sub_2939E80((__int64)&v84, a1, *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL));
  if ( !v83 )
    return 0;
  v4 = v87;
  if ( !v87 || v82 || v86 )
    return 0;
  v69 = *(_BYTE *)(*(_QWORD *)(v78 + 24) + 8LL) == 14;
  sub_23D0AB0((__int64)v103, a2, 0, 0, 0);
  sub_293CE40(v106, (_QWORD *)a1, a2, *(_QWORD *)(a2 - 32), &v84);
  v7 = v80;
  v8 = v102;
  v9 = v102;
  v10 = v80;
  v100 = v102;
  v101 = 0x800000000LL;
  if ( v80 )
  {
    if ( v80 > 8uLL )
    {
      v66 = v80;
      sub_C8D5F0((__int64)&v100, v102, v80, 8u, v80, v6);
      v9 = v100;
      v10 = v66;
      v8 = &v100[8 * (unsigned int)v101];
    }
    for ( i = &v9[8 * v7]; i != v8; ++v8 )
    {
      if ( v8 )
        *v8 = 0;
    }
    LODWORD(v101) = v10;
  }
  v12.m128i_i64[0] = sub_BCAE30((__int64)v81);
  v109 = v12;
  v64 = sub_CA1930(&v109);
  v13.m128i_i64[0] = sub_BCAE30(v85);
  v109 = v13;
  v14 = sub_CA1930(&v109);
  if ( v64 == v14 || v69 )
  {
    if ( v80 )
    {
      v65 = v4;
      for ( j = 0; j < v80; ++j )
      {
        v97.m128i_i32[0] = j;
        LOWORD(v99) = 265;
        v22.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v94 = v22;
        LOWORD(v96) = 773;
        v95.m128i_i64[0] = (__int64)".i";
        v17 = v99;
        if ( (_BYTE)v99 )
        {
          if ( (_BYTE)v99 == 1 )
          {
            v47 = _mm_loadu_si128(&v95);
            v109 = _mm_loadu_si128(&v94);
            v111 = v96;
            v110 = v47;
          }
          else
          {
            if ( BYTE1(v99) == 1 )
            {
              v67 = v97.m128i_i64[1];
              v16 = (__m128i *)v97.m128i_i64[0];
            }
            else
            {
              v16 = &v97;
              v17 = 2;
            }
            v110.m128i_i64[0] = (__int64)v16;
            LOBYTE(v111) = 2;
            v109.m128i_i64[0] = (__int64)&v94;
            BYTE1(v111) = v17;
            v110.m128i_i64[1] = v67;
          }
        }
        else
        {
          LOWORD(v111) = 256;
        }
        v18 = (__int64)v82;
        if ( !v82 || v80 - 1 != j )
          v18 = (__int64)v81;
        v71 = (__int64 **)v18;
        v19 = sub_293BC00((__int64)v106, j);
        v20 = sub_293B800(v103, 0x31u, v19, v71, (__int64)&v109, 0, v91.m128i_i32[0], 0);
        v21 = j;
        *(_QWORD *)&v100[8 * v21] = v20;
      }
      v4 = v65;
      v3 = a1;
    }
  }
  else if ( v14 % v64 )
  {
    if ( v64 % v14 )
    {
      v4 = 0;
      goto LABEL_64;
    }
    v75 = v64 / v14;
    v77 = 0;
    v74 = v84.m128i_i32[2];
    v73 = 0;
    v76 = 0;
    v23 = sub_BCDA70(*(__int64 **)(v84.m128i_i64[0] + 24), v64 / v14 * v84.m128i_i32[2]);
    v24 = v64 / v14;
    v73 = v23;
    v109.m128i_i64[1] = 0x800000000LL;
    v25 = v64 / v14;
    v76 = v85;
    v26 = &v110;
    v27 = &v110;
    v109.m128i_i64[0] = (__int64)&v110;
    if ( v64 / v14 )
    {
      if ( v75 > 8uLL )
      {
        sub_C8D5F0((__int64)&v109, &v110, v75, 8u, v75, v75);
        v27 = (__m128i *)v109.m128i_i64[0];
        v25 = v75;
        v24 = v75;
        v26 = (__m128i *)(v109.m128i_i64[0] + 8LL * v109.m128i_u32[2]);
      }
      for ( k = (__m128i *)((char *)v27 + 8 * v24); k != v26; v26 = (__m128i *)((char *)v26 + 8) )
      {
        if ( v26 )
          v26->m128i_i64[0] = 0;
      }
      v109.m128i_i32[2] = v25;
    }
    if ( v80 )
    {
      v29 = 0;
      v60 = v4;
      for ( m = 0; m < v80; ++m )
      {
        if ( v75 )
        {
          for ( n = 0; n < v75; ++n )
          {
            v32 = v29++;
            v33 = sub_293BC00((__int64)v106, v32);
            v34 = n;
            *(_QWORD *)(v109.m128i_i64[0] + 8 * v34) = v33;
          }
        }
        v94.m128i_i32[0] = m;
        LOWORD(v96) = 265;
        v35.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v91 = v35;
        LOWORD(v93) = 773;
        v92.m128i_i64[0] = (__int64)".i";
        v39 = v96;
        if ( (_BYTE)v96 )
        {
          if ( (_BYTE)v96 == 1 )
          {
            v59 = _mm_loadu_si128(&v92);
            v97 = _mm_loadu_si128(&v91);
            v99 = v93;
            v98 = v59;
          }
          else
          {
            if ( BYTE1(v96) == 1 )
            {
              v62 = v94.m128i_i64[1];
              v40 = (__m128i *)v94.m128i_i64[0];
            }
            else
            {
              v40 = &v94;
              v39 = 2;
            }
            v98.m128i_i64[0] = (__int64)v40;
            v36 = v62;
            v97.m128i_i64[0] = (__int64)&v91;
            v98.m128i_i64[1] = v62;
            LOBYTE(v99) = 2;
            BYTE1(v99) = v39;
          }
        }
        else
        {
          v37 = 256;
          LOWORD(v99) = 256;
        }
        v68 = sub_293ACB0(
                v103,
                v109.m128i_i64[0],
                (__int64)&v73,
                v36,
                v37,
                v38,
                (__int64 *)v97.m128i_i64[0],
                v97.m128i_i64[1],
                v98.m128i_i32[0],
                v98.m128i_i32[2],
                v99);
        LOWORD(v93) = 265;
        v91.m128i_i32[0] = m;
        v41.m128i_i64[0] = (__int64)sub_BD5D20(a2);
        v88 = v41;
        LOWORD(v90) = 773;
        v89.m128i_i64[0] = (__int64)".i";
        v42 = v93;
        if ( (_BYTE)v93 )
        {
          if ( (_BYTE)v93 == 1 )
          {
            v58 = _mm_loadu_si128(&v89);
            v94 = _mm_loadu_si128(&v88);
            v96 = v90;
            v95 = v58;
          }
          else
          {
            if ( BYTE1(v93) == 1 )
            {
              v61 = v91.m128i_i64[1];
              v43 = (__m128i *)v91.m128i_i64[0];
            }
            else
            {
              v43 = &v91;
              v42 = 2;
            }
            v95.m128i_i64[0] = (__int64)v43;
            v94.m128i_i64[0] = (__int64)&v88;
            v95.m128i_i64[1] = v61;
            LOBYTE(v96) = 2;
            BYTE1(v96) = v42;
          }
        }
        else
        {
          LOWORD(v96) = 256;
        }
        v44 = v82;
        if ( !v82 || v80 - 1 != m )
          v44 = v81;
        v45 = sub_293B800(v103, 0x31u, (unsigned __int64)v68, v44, (__int64)&v94, 0, v72, 0);
        v46 = m;
        *(_QWORD *)&v100[8 * v46] = v45;
      }
      v4 = v60;
      v3 = a1;
    }
    if ( (__m128i *)v109.m128i_i64[0] != &v110 )
      _libc_free(v109.m128i_u64[0]);
  }
  else
  {
    v97.m128i_i32[3] = v14 / v64;
    v97.m128i_i64[0] = 0;
    v97.m128i_i32[2] = v79;
    v98 = 0u;
    v97.m128i_i64[0] = sub_BCDA70(*(__int64 **)(v78 + 24), v14 / v64 * v79);
    v98.m128i_i64[0] = (__int64)v81;
    if ( v84.m128i_i32[3] )
    {
      v63 = v4;
      v48 = 0;
      v49 = 0;
      do
      {
        for ( ii = sub_293BC00((__int64)v106, v49); *(_BYTE *)ii == 78; ii = *v51 )
        {
          if ( (*(_BYTE *)(ii + 7) & 0x40) != 0 )
            v51 = *(__int64 **)(ii - 8);
          else
            v51 = (__int64 *)(ii - 32LL * (*(_DWORD *)(ii + 4) & 0x7FFFFFF));
        }
        v70 = ii;
        v52.m128i_i64[0] = (__int64)sub_BD5D20(ii);
        v109 = v52;
        v110.m128i_i64[0] = (__int64)".cast";
        LOWORD(v111) = 773;
        v53 = sub_293B800(v103, 0x31u, v70, (__int64 **)v97.m128i_i64[0], (__int64)&v109, 0, v94.m128i_i32[0], 0);
        sub_293CE40(&v109, (_QWORD *)a1, a2, v53, &v97);
        if ( v97.m128i_i32[3] )
        {
          v54 = 0;
          do
          {
            v55 = v54++;
            v56 = sub_293BC00((__int64)&v109, v55);
            v57 = v48++;
            *(_QWORD *)&v100[8 * v57] = v56;
          }
          while ( v97.m128i_i32[3] > v54 );
        }
        if ( v112 != &v113 )
          _libc_free((unsigned __int64)v112);
        ++v49;
      }
      while ( v84.m128i_i32[3] > v49 );
      v4 = v63;
      v3 = a1;
    }
  }
  sub_293CAB0(v3, a2, (__int64)&v100, (__int64)&v78);
LABEL_64:
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  if ( v107 != &v108 )
    _libc_free((unsigned __int64)v107);
  nullsub_61();
  v105 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v103[0] != &v104 )
    _libc_free(v103[0]);
  return v4;
}
