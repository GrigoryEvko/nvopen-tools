// Function: sub_B29620
// Address: 0xb29620
//
__int64 __fastcall sub_B29620(_QWORD **a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // r8
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 result; // rax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 *v22; // rsi
  __int64 *v23; // r8
  __int64 *v24; // rbx
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // r15
  __int64 v28; // rdx
  unsigned int v29; // eax
  unsigned int v30; // r13d
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r9
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 *v44; // rdi
  __int64 *v45; // rcx
  _QWORD **v46; // r13
  _QWORD **v47; // rbx
  _QWORD *v48; // rdi
  __int64 v49; // rbx
  _BYTE **v50; // rsi
  __int64 *v51; // rcx
  __int64 v52; // r11
  __int64 v53; // rcx
  __int64 v54; // r15
  __int64 v55; // r8
  __int64 i; // r9
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rsi
  __int64 v60; // r8
  __int64 v61; // rcx
  __int64 v62; // rsi
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rcx
  _QWORD *v66; // rdi
  _QWORD *v67; // rsi
  __int64 *v68; // rcx
  __int64 v69; // [rsp+8h] [rbp-2B8h]
  __int64 v70; // [rsp+38h] [rbp-288h]
  __int64 v71; // [rsp+38h] [rbp-288h]
  __int64 v72; // [rsp+38h] [rbp-288h]
  unsigned int v74; // [rsp+50h] [rbp-270h]
  unsigned int v75; // [rsp+54h] [rbp-26Ch]
  __int64 v76; // [rsp+58h] [rbp-268h] BYREF
  __int64 v77; // [rsp+68h] [rbp-258h] BYREF
  _BYTE v78[48]; // [rsp+70h] [rbp-250h] BYREF
  _BYTE *v79; // [rsp+A0h] [rbp-220h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-218h]
  _BYTE v81[64]; // [rsp+B0h] [rbp-210h] BYREF
  __int64 *v82; // [rsp+F0h] [rbp-1D0h] BYREF
  int v83; // [rsp+F8h] [rbp-1C8h]
  _BYTE v84[64]; // [rsp+100h] [rbp-1C0h] BYREF
  __int64 *v85; // [rsp+140h] [rbp-180h] BYREF
  __int64 v86; // [rsp+148h] [rbp-178h]
  _BYTE v87[72]; // [rsp+150h] [rbp-170h] BYREF
  __int64 v88; // [rsp+198h] [rbp-128h] BYREF
  __int64 v89; // [rsp+1A0h] [rbp-120h]
  __int64 v90; // [rsp+1A8h] [rbp-118h] BYREF
  unsigned int v91; // [rsp+1B0h] [rbp-110h]
  _BYTE *v92; // [rsp+1E8h] [rbp-D8h] BYREF
  __int64 v93; // [rsp+1F0h] [rbp-D0h]
  _BYTE v94[64]; // [rsp+1F8h] [rbp-C8h] BYREF
  _BYTE *v95; // [rsp+238h] [rbp-88h] BYREF
  __int64 v96; // [rsp+240h] [rbp-80h]
  _BYTE v97[120]; // [rsp+248h] [rbp-78h] BYREF

  v4 = a3;
  v6 = (__int64)a1;
  v7 = *(_QWORD **)(a4 + 8);
  v76 = a4;
  if ( !*v7 )
  {
    v66 = *a1;
    v85 = *(__int64 **)a4;
    v67 = &v66[*(unsigned int *)(v6 + 8)];
    if ( v67 != sub_B18540(v66, (__int64)v67, (__int64 *)&v85) )
      return sub_B28E70(v6, a2);
  }
  if ( *v4 && *(_QWORD *)a4 && (v8 = sub_B197A0(v6, *v4, *(_QWORD *)a4)) != 0 )
  {
    v9 = (unsigned int)(*(_DWORD *)(v8 + 44) + 1);
    v10 = *(_DWORD *)(v8 + 44) + 1;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  if ( *(_DWORD *)(v6 + 56) <= v10 )
LABEL_91:
    BUG();
  v69 = *(_QWORD *)(*(_QWORD *)(v6 + 48) + 8 * v9);
  result = (unsigned int)(*(_DWORD *)(v69 + 16) + 1);
  v75 = result;
  if ( (unsigned int)result >= *(_DWORD *)(a4 + 16) )
    return result;
  v88 = 0;
  v89 = 1;
  v85 = (__int64 *)v87;
  v86 = 0x800000000LL;
  v12 = &v90;
  do
    *v12++ = -4096;
  while ( v12 != (__int64 *)&v92 );
  v92 = v94;
  v95 = v97;
  v93 = 0x800000000LL;
  v96 = 0x800000000LL;
  v80 = 0x800000000LL;
  v79 = v81;
  sub_B1AE00((__int64)&v85, a4);
  v13 = (__int64)v85;
  v14 = (unsigned int)v86;
  v15 = v85[v14 - 1];
  v16 = ((v14 * 8) >> 3) - 1;
  v17 = (((v14 * 8) >> 3) - 2) / 2;
  if ( v16 > 0 )
  {
    while ( 1 )
    {
      v18 = v13 + 8 * v17;
      v68 = (__int64 *)(v13 + 8 * v16);
      if ( *(_DWORD *)(*(_QWORD *)v18 + 16LL) >= *(_DWORD *)(v15 + 16) )
        break;
      *v68 = *(_QWORD *)v18;
      v16 = v17;
      if ( v17 <= 0 )
      {
        v68 = (__int64 *)(v13 + 8 * v17);
        break;
      }
      v17 = (v17 - 1) / 2;
    }
  }
  else
  {
    v68 = &v85[v14 - 1];
  }
  *v68 = v15;
  sub_B24170((__int64)&v82, (__int64)&v88, &v76);
  v19 = (unsigned int)v86;
  if ( !(_DWORD)v86 )
    goto LABEL_51;
  do
  {
    v20 = (__int64)v85;
    v21 = *v85;
    if ( v19 == 1 )
      goto LABEL_19;
    v51 = &v85[v19];
    v52 = *(v51 - 1);
    *(v51 - 1) = v21;
    v53 = (__int64)v51 - v20 - 8;
    v54 = v53 >> 3;
    v55 = (v53 >> 3) & 1;
    if ( v53 > 16 )
    {
      for ( i = 0; ; i = v57 )
      {
        v57 = 2 * (i + 1);
        v58 = v20 + 16 * (i + 1);
        v59 = *(_QWORD *)v58;
        if ( *(_DWORD *)(*(_QWORD *)v58 + 16LL) < *(_DWORD *)(*(_QWORD *)(v58 - 8) + 16LL) )
        {
          --v57;
          v58 = v20 + 8 * v57;
          v59 = *(_QWORD *)v58;
        }
        *(_QWORD *)(v20 + 8 * i) = v59;
        if ( ((v53 >> 3) - 1) / 2 <= v57 )
          break;
      }
      if ( v55 )
      {
LABEL_85:
        v60 = v57;
        v61 = (v57 - 1) >> 1;
LABEL_79:
        while ( 1 )
        {
          v62 = v20 + 8 * v61;
          v63 = (_QWORD *)(v20 + 8 * v60);
          if ( *(_DWORD *)(*(_QWORD *)v62 + 16LL) >= *(_DWORD *)(v52 + 16) )
            goto LABEL_80;
          *v63 = *(_QWORD *)v62;
          v60 = v61;
          if ( !v61 )
          {
            *(_QWORD *)v62 = v52;
            goto LABEL_19;
          }
          v61 = (v61 - 1) / 2;
        }
      }
      v60 = v57;
      v61 = (v57 - 1) >> 1;
      if ( v57 != (v54 - 2) / 2 )
        goto LABEL_79;
LABEL_84:
      v64 = 2 * v57 + 2;
      v65 = *(_QWORD *)(v20 + 8 * v64 - 8);
      v57 = v64 - 1;
      *(_QWORD *)v58 = v65;
      goto LABEL_85;
    }
    v63 = (_QWORD *)v20;
    if ( !v55 && (unsigned __int64)((v53 >> 3) - 1) <= 2 )
    {
      v58 = v20;
      v57 = 0;
      goto LABEL_84;
    }
LABEL_80:
    *v63 = v52;
LABEL_19:
    LODWORD(v86) = v86 - 1;
    sub_B1AE00((__int64)&v92, v21);
    v74 = *(_DWORD *)(v21 + 16);
    while ( 2 )
    {
      v22 = *(__int64 **)v21;
      sub_B1CE50((__int64)&v82, *(_QWORD *)v21, a2);
      v23 = v82;
      v24 = &v82[v83];
      if ( v24 == v82 )
        goto LABEL_45;
      v25 = v6;
      v26 = v82;
      v27 = v25;
      do
      {
        while ( 1 )
        {
          v37 = *v26;
          if ( *v26 )
          {
            v28 = (unsigned int)(*(_DWORD *)(v37 + 44) + 1);
            v29 = *(_DWORD *)(v37 + 44) + 1;
          }
          else
          {
            v28 = 0;
            v29 = 0;
          }
          if ( v29 >= *(_DWORD *)(v27 + 56) )
          {
            v77 = 0;
            goto LABEL_91;
          }
          v77 = *(_QWORD *)(*(_QWORD *)(v27 + 48) + 8 * v28);
          v30 = *(_DWORD *)(v77 + 16);
          if ( v75 < v30 )
          {
            v22 = &v88;
            sub_B24170((__int64)v78, (__int64)&v88, &v77);
            if ( v78[32] )
              break;
          }
LABEL_32:
          if ( v24 == ++v26 )
            goto LABEL_44;
        }
        v31 = v77;
        if ( v74 < v30 )
        {
          v32 = (unsigned int)v80;
          v33 = (unsigned int)v80 + 1LL;
          if ( v33 > HIDWORD(v80) )
          {
            v22 = (__int64 *)v81;
            v71 = v77;
            sub_C8D5F0(&v79, v81, v33, 8);
            v32 = (unsigned int)v80;
            v31 = v71;
          }
          *(_QWORD *)&v79[8 * v32] = v31;
          v34 = (unsigned int)v96;
          LODWORD(v80) = v80 + 1;
          v35 = (unsigned int)v96 + 1LL;
          v36 = v77;
          if ( v35 > HIDWORD(v96) )
          {
            v22 = (__int64 *)v97;
            v70 = v77;
            sub_C8D5F0(&v95, v97, v35, 8);
            v34 = (unsigned int)v96;
            v36 = v70;
          }
          *(_QWORD *)&v95[8 * v34] = v36;
          LODWORD(v96) = v96 + 1;
          goto LABEL_32;
        }
        v38 = (unsigned int)v86;
        v39 = (unsigned int)v86 + 1LL;
        if ( v39 > HIDWORD(v86) )
        {
          v72 = v77;
          sub_C8D5F0(&v85, v87, v39, 8);
          v38 = (unsigned int)v86;
          v31 = v72;
        }
        v85[v38] = v31;
        v22 = v85;
        LODWORD(v86) = v86 + 1;
        v40 = (unsigned int)v86;
        v41 = v85[v40 - 1];
        v42 = ((v40 * 8) >> 3) - 1;
        v43 = (((v40 * 8) >> 3) - 2) / 2;
        if ( v42 > 0 )
        {
          while ( 1 )
          {
            v44 = &v22[v43];
            v45 = &v22[v42];
            if ( *(_DWORD *)(*v44 + 16) >= *(_DWORD *)(v41 + 16) )
            {
              *v45 = v41;
              goto LABEL_43;
            }
            *v45 = *v44;
            v42 = v43;
            if ( v43 <= 0 )
              break;
            v43 = (v43 - 1) / 2;
          }
          *v44 = v41;
        }
        else
        {
          v85[v40 - 1] = v41;
        }
LABEL_43:
        ++v26;
      }
      while ( v24 != v26 );
LABEL_44:
      v23 = v82;
      v6 = v27;
LABEL_45:
      if ( v23 != (__int64 *)v84 )
        _libc_free(v23, v22);
      if ( (_DWORD)v80 )
      {
        v21 = *(_QWORD *)&v79[8 * (unsigned int)v80 - 8];
        LODWORD(v80) = v80 - 1;
        continue;
      }
      break;
    }
    v19 = (unsigned int)v86;
  }
  while ( (_DWORD)v86 );
LABEL_51:
  v46 = (_QWORD **)v92;
  v47 = (_QWORD **)&v92[8 * (unsigned int)v93];
  if ( v92 != (_BYTE *)v47 )
  {
    do
    {
      v48 = *v46++;
      sub_B1AE50(v48, v69);
    }
    while ( v47 != v46 );
  }
  v49 = *(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8);
  v50 = (_BYTE **)v49;
  result = (__int64)sub_B1D3D0(*(__int64 **)v6, v49, a2);
  if ( v49 != result )
  {
    sub_B28710(&v82, v6, a2);
    v50 = (_BYTE **)&v82;
    if ( !(unsigned __int8)sub_B1B2E0(v6, (__int64)&v82) )
    {
      v50 = (_BYTE **)a2;
      sub_B28E70(v6, a2);
    }
    result = (__int64)v84;
    if ( v82 != (__int64 *)v84 )
      result = _libc_free(v82, v50);
  }
  if ( v79 != v81 )
    result = _libc_free(v79, v50);
  if ( v95 != v97 )
    result = _libc_free(v95, v50);
  if ( v92 != v94 )
    result = _libc_free(v92, v50);
  if ( (v89 & 1) == 0 )
  {
    v50 = (_BYTE **)(8LL * v91);
    result = sub_C7D6A0(v90, v50, 8);
  }
  if ( v85 != (__int64 *)v87 )
    return _libc_free(v85, v50);
  return result;
}
