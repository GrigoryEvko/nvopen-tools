// Function: sub_2D3B6F0
// Address: 0x2d3b6f0
//
void __fastcall sub_2D3B6F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // r13
  __m128i v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rax
  __m128i v17; // xmm0
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  char v20; // di
  unsigned int v21; // eax
  __m128i *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rcx
  unsigned __int32 v25; // r13d
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // esi
  __int64 v30; // rcx
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // r9
  unsigned int v35; // esi
  unsigned __int32 *v36; // rdx
  __int64 v37; // r8
  unsigned int v38; // eax
  unsigned int v39; // esi
  __int64 v40; // r8
  __int64 v41; // r9
  __m128i *v42; // r15
  unsigned __int32 v43; // eax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdi
  int v51; // esi
  int *v52; // rax
  int *v53; // r15
  int *v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned __int64 v59; // r8
  unsigned __int64 v60; // r9
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rsi
  __int64 v64; // rdx
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  int *v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r8
  __int64 v71; // r9
  int v72; // edx
  int v73; // r10d
  __int64 v74; // [rsp-10h] [rbp-240h]
  __int64 v75; // [rsp-8h] [rbp-238h]
  int *v76; // [rsp+8h] [rbp-228h]
  unsigned __int32 v77; // [rsp+28h] [rbp-208h]
  unsigned __int32 v78; // [rsp+28h] [rbp-208h]
  __int8 v79; // [rsp+2Ch] [rbp-204h]
  unsigned int v80; // [rsp+2Ch] [rbp-204h]
  unsigned int v81; // [rsp+30h] [rbp-200h]
  _DWORD *v85; // [rsp+48h] [rbp-1E8h]
  _QWORD *v86; // [rsp+50h] [rbp-1E0h]
  unsigned int v87; // [rsp+50h] [rbp-1E0h]
  _QWORD *v88; // [rsp+58h] [rbp-1D8h]
  __int64 v89; // [rsp+58h] [rbp-1D8h]
  unsigned __int32 v90; // [rsp+58h] [rbp-1D8h]
  _QWORD *v91; // [rsp+58h] [rbp-1D8h]
  unsigned int v92; // [rsp+6Ch] [rbp-1C4h] BYREF
  __int64 v93; // [rsp+70h] [rbp-1C0h] BYREF
  _BYTE *v94; // [rsp+78h] [rbp-1B8h] BYREF
  __int64 v95; // [rsp+80h] [rbp-1B0h]
  _BYTE v96[72]; // [rsp+88h] [rbp-1A8h] BYREF
  _DWORD *v97; // [rsp+D0h] [rbp-160h] BYREF
  _BYTE *v98; // [rsp+D8h] [rbp-158h]
  __int64 v99; // [rsp+E0h] [rbp-150h]
  _BYTE v100[72]; // [rsp+E8h] [rbp-148h] BYREF
  __m128i v101[12]; // [rsp+130h] [rbp-100h] BYREF
  __int64 v102; // [rsp+1F0h] [rbp-40h]
  _QWORD *v103; // [rsp+1F8h] [rbp-38h]

  v7 = (__int64 *)(*(_QWORD *)(a1[1] + 48LL) + 40LL * (unsigned int)(*(_DWORD *)a2 - 1));
  v8 = *v7;
  v9 = v7[4];
  v10.m128i_i64[0] = sub_AF3FE0(*v7);
  v101[0] = v10;
  v79 = v10.m128i_i8[8];
  if ( !v10.m128i_i8[8] )
    return;
  v11 = a1[2];
  v101[0].m128i_i64[0] = v8;
  v101[0].m128i_i64[1] = v9;
  if ( !sub_2D2BDF0(v11, v101[0].m128i_i64) )
    return;
  v12 = sub_B10D40(a2 + 16);
  v101[0].m128i_i64[0] = v8;
  v101[0].m128i_i64[1] = v12;
  v13 = v12;
  v14 = (unsigned __int64)(a1 + 27);
  v15 = (_QWORD *)a1[28];
  if ( !v15 )
    goto LABEL_12;
  do
  {
    while ( v8 <= v15[4] && (v8 != v15[4] || v13 <= v15[5]) )
    {
      v14 = (unsigned __int64)v15;
      v15 = (_QWORD *)v15[2];
      if ( !v15 )
        goto LABEL_10;
    }
    v15 = (_QWORD *)v15[3];
  }
  while ( v15 );
LABEL_10:
  if ( a1 + 27 == (_QWORD *)v14
    || v8 < *(_QWORD *)(v14 + 32)
    || v8 == *(_QWORD *)(v14 + 32) && v13 < *(_QWORD *)(v14 + 40) )
  {
LABEL_12:
    v88 = (_QWORD *)v14;
    v86 = a1 + 27;
    v16 = sub_22077B0(0x38u);
    v17 = _mm_loadu_si128(v101);
    *(_DWORD *)(v16 + 48) = 0;
    v14 = v16;
    *(__m128i *)(v16 + 32) = v17;
    v18 = sub_2D2D900(a1 + 26, v88, (unsigned __int64 *)(v16 + 32));
    if ( v19 )
    {
      v20 = 1;
      if ( !v18 && v86 != v19 )
      {
        v66 = v19[4];
        if ( *(_QWORD *)(v14 + 32) >= v66 )
        {
          v20 = 0;
          if ( *(_QWORD *)(v14 + 32) == v66 )
            v20 = *(_QWORD *)(v14 + 40) < v19[5];
        }
      }
      sub_220F040(v20, v14, v19, v86);
      ++a1[31];
    }
    else
    {
      v91 = v18;
      j_j___libc_free_0(v14);
      v14 = (unsigned __int64)v91;
    }
  }
  v21 = *(_DWORD *)(v14 + 48);
  if ( !v21 )
  {
    *(_DWORD *)(v14 + 48) = ((__int64)(a1[33] - a1[32]) >> 4) + 1;
    v22 = (__m128i *)a1[33];
    if ( v22 == (__m128i *)a1[34] )
    {
      sub_2D2ABA0(a1 + 32, v22, v101);
    }
    else
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v101);
        v22 = (__m128i *)a1[33];
      }
      a1[33] = v22 + 1;
    }
    v21 = *(_DWORD *)(v14 + 48);
  }
  v23 = *(_QWORD *)(a2 + 8);
  v92 = v21;
  v89 = v23;
  sub_AF47B0((__int64)v101, *(unsigned __int64 **)(v23 + 16), *(unsigned __int64 **)(v23 + 24));
  v24 = v89;
  if ( v101[1].m128i_i8[0] )
  {
    v25 = v101[0].m128i_u32[2];
    v90 = v101[0].m128i_i32[0] + v101[0].m128i_i32[2];
  }
  else
  {
    v25 = 0;
    v55 = sub_AF3FE0(v8);
    v24 = v89;
    v97 = (_DWORD *)v55;
    v98 = (_BYTE *)v56;
    v90 = v55;
  }
  v26 = *(__int64 **)(v24 + 16);
  v27 = (__int64)(*(_QWORD *)(v24 + 24) - (_QWORD)v26) >> 3;
  v87 = v27;
  if ( (unsigned int)v27 <= 2 )
  {
    if ( !(_DWORD)v27 )
      goto LABEL_35;
    v50 = 0;
    v30 = 0;
    v51 = 0;
    goto LABEL_74;
  }
  v28 = *v26;
  if ( *v26 == 35 )
  {
    v30 = v26[1];
    v50 = 2;
    v51 = 2;
LABEL_74:
    v87 = 0;
    if ( v26[v50] != 6 )
      goto LABEL_35;
    v31 = (unsigned int)(v51 + 1);
    if ( (_DWORD)v27 == (_DWORD)v31 )
    {
LABEL_32:
      if ( 8 * v30 == v25 )
        v87 = sub_2D2DBF0(a1 + 17, (unsigned __int64 *)(a2 + 24));
      else
        v87 = 0;
      goto LABEL_35;
    }
LABEL_76:
    v29 = v51 + 4;
    goto LABEL_30;
  }
  if ( (_DWORD)v27 == 3 )
  {
    v87 = 0;
    if ( *v26 != 6 )
      goto LABEL_35;
    v30 = 0;
    v51 = 0;
    v31 = 1;
    goto LABEL_76;
  }
  if ( v28 == 16 )
  {
    v65 = v26[2];
    if ( v65 == 34 )
    {
      v30 = v26[1];
      v50 = 3;
      v51 = 3;
    }
    else
    {
      v87 = 0;
      if ( v65 != 28 )
        goto LABEL_35;
      v50 = 3;
      v51 = 3;
      v30 = -v26[1];
    }
    goto LABEL_74;
  }
  v87 = 0;
  if ( v28 == 6 )
  {
    v29 = 4;
    v30 = 0;
    v31 = 1;
LABEL_30:
    v87 = 0;
    if ( (_DWORD)v27 != v29 || v26[v31] != 4096 )
      goto LABEL_35;
    goto LABEL_32;
  }
LABEL_35:
  v32 = *(_QWORD *)(a5 + 8);
  v33 = *(unsigned int *)(a5 + 24);
  if ( !(_DWORD)v33 )
    goto LABEL_87;
  v34 = (unsigned int)(v33 - 1);
  v35 = v34 & (37 * v92);
  v36 = (unsigned __int32 *)(v32 + 216LL * v35);
  v37 = *v36;
  if ( v92 != (_DWORD)v37 )
  {
    v72 = 1;
    while ( (_DWORD)v37 != -1 )
    {
      v73 = v72 + 1;
      v35 = v34 & (v72 + v35);
      v36 = (unsigned __int32 *)(v32 + 216LL * v35);
      v37 = *v36;
      if ( v92 == (_DWORD)v37 )
        goto LABEL_37;
      v72 = v73;
    }
    goto LABEL_87;
  }
LABEL_37:
  if ( v36 == (unsigned __int32 *)(v32 + 216 * v33) )
  {
LABEL_87:
    v103 = a1 + 4;
    v102 = 0;
    memset(v101, 0, sizeof(v101));
    sub_2D2E0E0((__int64)&v97, a5, (int *)&v92, (__int64)v101);
    sub_2D2A900((__int64)v101, a5, v57, v58, v59, v60);
    sub_2D35160(v99 + 8, v25, v90, v87, v61, v62);
    return;
  }
  v101[0].m128i_i64[0] = (__int64)(v36 + 2);
  v85 = v36 + 2;
  v101[0].m128i_i64[1] = (__int64)&v101[1].m128i_i64[1];
  v101[1].m128i_i64[0] = 0x400000000LL;
  v38 = v36[50];
  if ( v38 )
  {
    sub_2D2BC70((__int64)v101, v25, (__int64)v36, v92, v37, v34);
  }
  else
  {
    v39 = v36[51];
    if ( v39 )
    {
      v36 += 3;
      while ( v25 >= *v36 )
      {
        ++v38;
        v36 += 2;
        if ( v39 == v38 )
          goto LABEL_44;
      }
      v39 = v38;
    }
LABEL_44:
    sub_2D29C80((__int64)v101, v39, (__int64)v36, v92, v37, v34);
  }
  v42 = (__m128i *)v101[0].m128i_i64[1];
  if ( v101[1].m128i_i32[0] && *(_DWORD *)(v101[0].m128i_i64[1] + 12) < *(_DWORD *)(v101[0].m128i_i64[1] + 8) )
  {
    v43 = *(_DWORD *)sub_2D289F0((__int64)v101);
    if ( v42 != (__m128i *)&v101[1].m128i_u64[1] )
    {
      v77 = v43;
      _libc_free((unsigned __int64)v42);
      v43 = v77;
    }
    if ( v90 > v43 )
    {
      v94 = v96;
      v95 = 0x400000000LL;
      v93 = (__int64)v85;
      sub_2D2BCF0((__int64)&v93, v25, 0x400000000LL, (__int64)v96, v40, v41);
      v44 = *(unsigned int *)sub_2D289F0((__int64)&v93);
      v98 = v100;
      v97 = v85;
      v78 = v44;
      v99 = 0x400000000LL;
      sub_2D2BCF0((__int64)&v97, v90, 0x400000000LL, v44, v45, v46);
      if ( (_DWORD)v99 && *((_DWORD *)v98 + 3) < *((_DWORD *)v98 + 2) && *(_DWORD *)sub_2D289F0((__int64)&v97) < v90 )
      {
        if ( v78 >= v25 )
          goto LABEL_82;
        if ( sub_2D28840((__int64)&v93, (__int64)&v97) )
        {
          v81 = *(_DWORD *)sub_2D28A10((__int64)&v93);
          v80 = *(_DWORD *)sub_2D28A30((__int64)&v93);
          sub_2D2B360((__int64)&v93, v25);
          v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
          if ( v101[0].m128i_i64[0] )
            sub_2D23AB0(v101[0].m128i_i64);
          v67 = (int *)sub_2D289F0((__int64)&v93);
          sub_2D3AEE0((__int64)a1, a4, a3, v92, *v67, v25, v80, v101[0].m128i_i64);
          sub_9C6650(v101);
          sub_2D35160((__int64)v85, v90, v81, v80, v68, v69);
          v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
          if ( v101[0].m128i_i64[0] )
            sub_2D23AB0(v101[0].m128i_i64);
          sub_2D3AEE0((__int64)a1, a4, a3, v92, v90, v81, v80, v101[0].m128i_i64);
          sub_9C6650(v101);
          sub_2D35160((__int64)v85, v25, v90, v87, v70, v71);
          goto LABEL_58;
        }
      }
      else
      {
        if ( v78 >= v25 )
          goto LABEL_54;
        v79 = 0;
      }
      sub_2D2B360((__int64)&v93, v25);
      v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
      if ( v101[0].m128i_i64[0] )
        sub_2D23AB0(v101[0].m128i_i64);
      v76 = (int *)sub_2D28A30((__int64)&v93);
      v52 = (int *)sub_2D289F0((__int64)&v93);
      sub_2D3AEE0((__int64)a1, a4, a3, v92, *v52, v25, *v76, v101[0].m128i_i64);
      sub_9C6650(v101);
      v48 = v75;
      if ( !v79 )
      {
LABEL_54:
        v101[0].m128i_i64[0] = v93;
        v101[0].m128i_i64[1] = (__int64)&v101[1].m128i_i64[1];
        v101[1].m128i_i64[0] = 0x400000000LL;
        if ( (_DWORD)v95 )
        {
          sub_2D23820((__int64)&v101[0].m128i_i64[1], (__int64)&v94, (unsigned int)v95, v47, v48, v49);
          if ( v78 >= v25 )
            goto LABEL_91;
        }
        else if ( v78 >= v25 )
        {
LABEL_56:
          sub_2D35160((__int64)v85, v25, v90, v87, v48, v49);
          if ( (unsigned __int64 *)v101[0].m128i_i64[1] != &v101[1].m128i_u64[1] )
            _libc_free(v101[0].m128i_u64[1]);
LABEL_58:
          v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
          if ( v101[0].m128i_i64[0] )
            sub_2D23AB0(v101[0].m128i_i64);
          sub_2D3B550((__int64)a1, a4, a3, v92, v25, v90, v87, v101[0].m128i_i64, v85);
          sub_9C6650(v101);
          if ( v98 != v100 )
            _libc_free((unsigned __int64)v98);
          if ( v94 != v96 )
            _libc_free((unsigned __int64)v94);
          return;
        }
        sub_2D23A60((__int64)v101);
LABEL_91:
        while ( v101[1].m128i_i32[0] )
        {
          v63 = *(unsigned int *)(v101[0].m128i_i64[1] + 8);
          if ( *(_DWORD *)(v101[0].m128i_i64[1] + 12) >= (unsigned int)v63 )
            break;
          if ( *(_DWORD *)sub_2D289F0((__int64)v101) < v25 )
            break;
          if ( *(_DWORD *)sub_2D28A10((__int64)v101) > v90 )
            break;
          sub_2D2B2D0((__int64)v101, v63, v64, v90, v48, v49);
        }
        goto LABEL_56;
      }
LABEL_82:
      sub_2D2B410((unsigned int *)&v97, v90);
      v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
      if ( v101[0].m128i_i64[0] )
        sub_2D23AB0(v101[0].m128i_i64);
      v53 = (int *)sub_2D28A30((__int64)&v97);
      v54 = (int *)sub_2D28A10((__int64)&v97);
      sub_2D3AEE0((__int64)a1, a4, a3, v92, v90, *v54, *v53, v101[0].m128i_i64);
      sub_9C6650(v101);
      v47 = v74;
      goto LABEL_54;
    }
  }
  else if ( (unsigned __int64 *)v101[0].m128i_i64[1] != &v101[1].m128i_u64[1] )
  {
    _libc_free(v101[0].m128i_u64[1]);
  }
  sub_2D35160((__int64)v85, v25, v90, v87, v40, v41);
  v101[0].m128i_i64[0] = *(_QWORD *)(a2 + 16);
  if ( v101[0].m128i_i64[0] )
    sub_2D23AB0(v101[0].m128i_i64);
  sub_2D3B550((__int64)a1, a4, a3, v92, v25, v90, v87, v101[0].m128i_i64, v85);
  sub_9C6650(v101);
}
