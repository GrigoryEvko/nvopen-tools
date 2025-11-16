// Function: sub_2DF0160
// Address: 0x2df0160
//
__int64 __fastcall sub_2DF0160(unsigned __int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // r14
  __int64 v6; // rcx
  __int64 v8; // r15
  char v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned int v18; // r12d
  unsigned int v19; // ebx
  __int64 v20; // r13
  int *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r13
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rdi
  __int64 v33; // r15
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // rdi
  __int64 v36; // r14
  _QWORD *i; // rbx
  char v38; // r15
  __int64 v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r14
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // rbx
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rbx
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rbx
  unsigned __int64 v51; // rdi
  __int64 v52; // r14
  _QWORD *v53; // r12
  _QWORD *j; // rbx
  char v55; // r15
  __int64 v56; // rax
  _QWORD *v57; // rax
  _QWORD *v58; // rdx
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  _QWORD *v61; // r12
  char v62; // r13
  _QWORD *v63; // r14
  __int64 v64; // rax
  unsigned int v65; // [rsp+10h] [rbp-240h]
  unsigned int v66; // [rsp+14h] [rbp-23Ch]
  int *v68; // [rsp+20h] [rbp-230h]
  unsigned __int8 v69; // [rsp+2Bh] [rbp-225h]
  int v70; // [rsp+2Ch] [rbp-224h]
  __int64 v71; // [rsp+30h] [rbp-220h]
  unsigned int v72; // [rsp+40h] [rbp-210h]
  unsigned int v73; // [rsp+44h] [rbp-20Ch]
  __int64 v74; // [rsp+50h] [rbp-200h]
  _QWORD *v75; // [rsp+50h] [rbp-200h]
  _QWORD *v76; // [rsp+50h] [rbp-200h]
  char v77; // [rsp+58h] [rbp-1F8h]
  _QWORD v78[4]; // [rsp+60h] [rbp-1F0h] BYREF
  char v79; // [rsp+80h] [rbp-1D0h] BYREF
  unsigned __int64 v80; // [rsp+88h] [rbp-1C8h]
  __int64 v81; // [rsp+90h] [rbp-1C0h]
  char v82; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned __int64 v83; // [rsp+B8h] [rbp-198h]
  __int64 v84; // [rsp+C0h] [rbp-190h]
  __int64 v85; // [rsp+E0h] [rbp-170h]
  int v86; // [rsp+F0h] [rbp-160h]
  __int64 v87; // [rsp+F8h] [rbp-158h]
  _BYTE *v88; // [rsp+100h] [rbp-150h] BYREF
  __int64 v89; // [rsp+108h] [rbp-148h]
  _BYTE v90[96]; // [rsp+110h] [rbp-140h] BYREF
  unsigned __int64 v91; // [rsp+170h] [rbp-E0h] BYREF
  unsigned int v92; // [rsp+178h] [rbp-D8h]
  unsigned __int64 v93; // [rsp+180h] [rbp-D0h] BYREF
  __int64 v94; // [rsp+188h] [rbp-C8h]
  _BYTE *v95; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v96; // [rsp+198h] [rbp-B8h]
  _BYTE v97[96]; // [rsp+1A0h] [rbp-B0h] BYREF
  unsigned __int64 v98; // [rsp+200h] [rbp-50h] BYREF
  unsigned int v99; // [rsp+208h] [rbp-48h]
  __int64 v100; // [rsp+210h] [rbp-40h]

  v3 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v3 <= 0x1Cu )
    return 0;
  v4 = *(_QWORD *)(v3 + 8);
  if ( *(_BYTE *)(v4 + 8) != 17 )
    return 0;
  v6 = a2[17];
  v66 = *(_DWORD *)(v4 + 32);
  v65 = *(_DWORD *)(v6 + 32);
  v72 = v65 / v66;
  if ( v65 % v66 )
    return 0;
  v8 = *(_QWORD *)(v6 + 24);
  v9 = sub_AE5020(a3, v8);
  v10 = sub_9208B0(a3, v8);
  v94 = v11;
  v93 = ((1LL << v9) + ((unsigned __int64)(v10 + 7) >> 3) - 1) >> v9 << v9;
  v70 = sub_CA1930(&v93);
  v74 = *(_QWORD *)(v4 + 24);
  v77 = sub_AE5020(a3, v74);
  v12 = sub_9208B0(a3, v74);
  v94 = v13;
  v93 = ((1LL << v77) + ((unsigned __int64)(v12 + 7) >> 3) - 1) >> v77 << v77;
  if ( v72 * v70 != (unsigned int)sub_CA1930(&v93) )
    return 0;
  sub_2DEB2E0((__int64)v78, v4);
  v14 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 <= 0x1Cu )
    goto LABEL_60;
  switch ( v14 )
  {
    case '\\':
      v69 = sub_2DEF330(v3, v78, a3);
      break;
    case '=':
      v69 = sub_2DEC8F0((unsigned __int8 *)v3, v78, a3);
      break;
    case 'N':
      v69 = sub_2DF0160(v3, v78, a3);
      break;
    default:
      goto LABEL_60;
  }
  if ( v69 )
  {
    v73 = 0;
    v17 = a2[17];
    if ( *(_DWORD *)(v17 + 32) )
    {
      do
      {
        if ( v66 <= v65 )
        {
          v18 = 0;
          v19 = 0;
          v71 = 152LL * (v73 / v72);
          do
          {
            v20 = 0;
            v21 = (int *)(v85 + v71);
            if ( !v19 )
              v20 = *((_QWORD *)v21 + 18);
            v86 = *v21;
            v22 = *((_QWORD *)v21 + 1);
            v88 = v90;
            v87 = v22;
            v89 = 0x400000000LL;
            v23 = (unsigned int)v21[6];
            if ( (_DWORD)v23 )
            {
              v68 = (int *)(v85 + v71);
              sub_2DEB050((__int64)&v88, (__int64 *)v21 + 2, v23, 0x400000000LL, v15, v16);
              v21 = v68;
            }
            v92 = v21[34];
            if ( v92 > 0x40 )
              sub_C43780((__int64)&v91, (const void **)v21 + 16);
            else
              v91 = *((_QWORD *)v21 + 16);
            sub_C46A40((__int64)&v91, v18);
            LODWORD(v93) = v86;
            v94 = v87;
            v95 = v97;
            v96 = 0x400000000LL;
            if ( (_DWORD)v89 )
              sub_2DEB050((__int64)&v95, (__int64 *)&v88, v24, v25, v26, v27);
            v99 = v92;
            if ( v92 > 0x40 )
              sub_C43780((__int64)&v98, (const void **)&v91);
            else
              v98 = v91;
            v100 = v20;
            v28 = a2[16] + 152LL * (v19 + v73);
            *(_DWORD *)v28 = v93;
            *(_QWORD *)(v28 + 8) = v94;
            sub_2DEB400(v28 + 16, (unsigned __int64 *)&v95, 19LL * (v19 + v73), v25, v26, v27);
            if ( *(_DWORD *)(v28 + 136) > 0x40u )
            {
              v29 = *(_QWORD *)(v28 + 128);
              if ( v29 )
                j_j___libc_free_0_0(v29);
            }
            *(_QWORD *)(v28 + 128) = v98;
            *(_DWORD *)(v28 + 136) = v99;
            *(_QWORD *)(v28 + 144) = v100;
            v30 = (unsigned __int64)v95;
            v31 = (unsigned __int64)&v95[24 * (unsigned int)v96];
            if ( v95 != (_BYTE *)v31 )
            {
              do
              {
                v31 -= 24LL;
                if ( *(_DWORD *)(v31 + 16) > 0x40u )
                {
                  v32 = *(_QWORD *)(v31 + 8);
                  if ( v32 )
                    j_j___libc_free_0_0(v32);
                }
              }
              while ( v30 != v31 );
              v31 = (unsigned __int64)v95;
            }
            if ( (_BYTE *)v31 != v97 )
              _libc_free(v31);
            if ( v92 > 0x40 && v91 )
              j_j___libc_free_0_0(v91);
            v33 = (__int64)v88;
            v34 = (unsigned __int64)&v88[24 * (unsigned int)v89];
            if ( v88 != (_BYTE *)v34 )
            {
              do
              {
                v34 -= 24LL;
                if ( *(_DWORD *)(v34 + 16) > 0x40u )
                {
                  v35 = *(_QWORD *)(v34 + 8);
                  if ( v35 )
                    j_j___libc_free_0_0(v35);
                }
              }
              while ( v33 != v34 );
              v34 = (unsigned __int64)v88;
            }
            if ( (_BYTE *)v34 != v90 )
              _libc_free(v34);
            ++v19;
            v18 += v70;
          }
          while ( v72 > v19 );
          v17 = a2[17];
        }
        v73 += v72;
      }
      while ( v73 < *(_DWORD *)(v17 + 32) );
    }
    v36 = v81;
    a2[1] = v78[1];
    a2[2] = v78[2];
    for ( i = a2 + 4; (char *)v36 != &v79; v36 = sub_220EF30(v36) )
    {
      v40 = sub_2DEDBB0(a2 + 3, (__int64)i, (unsigned __int64 *)(v36 + 32));
      if ( v41 )
      {
        v38 = v40 || v41 == i || *(_QWORD *)(v36 + 32) < v41[4];
        v75 = v41;
        v39 = sub_22077B0(0x28u);
        *(_QWORD *)(v39 + 32) = *(_QWORD *)(v36 + 32);
        sub_220F040(v38, v39, v75, i);
        ++a2[8];
      }
    }
    v52 = v84;
    v53 = a2 + 9;
    for ( j = a2 + 10; (char *)v52 != &v82; v52 = sub_220EF30(v52) )
    {
      v57 = sub_2D12A40(v53, (__int64)j, (unsigned __int64 *)(v52 + 32));
      if ( v58 )
      {
        v55 = v57 || v58 == j || *(_QWORD *)(v52 + 32) < v58[4];
        v76 = v58;
        v56 = sub_22077B0(0x28u);
        *(_QWORD *)(v56 + 32) = *(_QWORD *)(v52 + 32);
        sub_220F040(v55, v56, v76, j);
        ++a2[14];
      }
    }
    v93 = a1;
    v59 = sub_2D11AF0((__int64)v53, &v93);
    v61 = v60;
    if ( v60 )
    {
      v62 = 1;
      v63 = a2 + 10;
      if ( !v59 && v63 != v60 )
        v62 = a1 < v60[4];
      v64 = sub_22077B0(0x28u);
      *(_QWORD *)(v64 + 32) = v93;
      sub_220F040(v62, v64, v61, v63);
      ++a2[14];
    }
    a2[15] = 0;
    goto LABEL_61;
  }
LABEL_60:
  v69 = 0;
LABEL_61:
  v78[0] = off_49D4228;
  if ( v85 )
  {
    v42 = 152LL * *(_QWORD *)(v85 - 8);
    v43 = v85 + v42;
    while ( v85 != v43 )
    {
      v43 -= 152;
      if ( *(_DWORD *)(v43 + 136) > 0x40u )
      {
        v44 = *(_QWORD *)(v43 + 128);
        if ( v44 )
          j_j___libc_free_0_0(v44);
      }
      v45 = *(_QWORD *)(v43 + 16);
      v46 = v45 + 24LL * *(unsigned int *)(v43 + 24);
      if ( v45 != v46 )
      {
        do
        {
          v46 -= 24LL;
          if ( *(_DWORD *)(v46 + 16) > 0x40u )
          {
            v47 = *(_QWORD *)(v46 + 8);
            if ( v47 )
              j_j___libc_free_0_0(v47);
          }
        }
        while ( v45 != v46 );
        v45 = *(_QWORD *)(v43 + 16);
      }
      if ( v45 != v43 + 32 )
        _libc_free(v45);
    }
    j_j_j___libc_free_0_0(v43 - 8);
  }
  v48 = v83;
  while ( v48 )
  {
    sub_2DEAE80(*(_QWORD *)(v48 + 24));
    v49 = v48;
    v48 = *(_QWORD *)(v48 + 16);
    j_j___libc_free_0(v49);
  }
  v50 = v80;
  while ( v50 )
  {
    sub_2DEACB0(*(_QWORD *)(v50 + 24));
    v51 = v50;
    v50 = *(_QWORD *)(v50 + 16);
    j_j___libc_free_0(v51);
  }
  return v69;
}
