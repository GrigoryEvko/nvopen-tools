// Function: sub_25D56C0
// Address: 0x25d56c0
//
__int64 __fastcall sub_25D56C0(__int64 a1)
{
  __int64 v2; // rax
  char *v3; // rcx
  const char **v4; // rdx
  __int64 v5; // rax
  char v6; // dl
  char *v7; // r12
  __int64 v8; // r14
  unsigned int v9; // eax
  __int64 v10; // r9
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned int v18; // ecx
  char *v19; // rdi
  __int64 v20; // rsi
  unsigned int v21; // edx
  int v22; // edx
  unsigned __int64 v23; // rdx
  int v24; // r15d
  __int64 v25; // r13
  _QWORD *v26; // rax
  _QWORD *j; // rdx
  const void *v28; // r15
  size_t v29; // r13
  int v30; // eax
  unsigned int v31; // r9d
  __int64 *v32; // r10
  __int64 i; // rcx
  __int64 v34; // rcx
  unsigned __int64 *v35; // r12
  __int64 v36; // r14
  unsigned __int64 *v37; // rbx
  __int64 v38; // r13
  __int64 v39; // r9
  unsigned __int64 v40; // r10
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rcx
  void *v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // r8
  _BYTE *v48; // r13
  _BYTE *v49; // r14
  unsigned __int64 v50; // rdi
  __int64 v51; // r13
  unsigned __int64 v52; // rbx
  volatile signed __int32 *v53; // r12
  signed __int32 v54; // eax
  signed __int32 v55; // eax
  __int64 v56; // rbx
  unsigned __int64 v57; // r12
  volatile signed __int32 *v58; // r13
  signed __int32 v59; // eax
  signed __int32 v60; // eax
  __int64 result; // rax
  __int64 v62; // rax
  unsigned int v63; // r9d
  __int64 *v64; // r10
  __int64 v65; // rcx
  __int64 *v66; // rax
  char *v67; // rsi
  __int64 v68; // [rsp+0h] [rbp-2E0h]
  __int64 *v69; // [rsp+8h] [rbp-2D8h]
  __int64 v70; // [rsp+10h] [rbp-2D0h]
  unsigned int v71; // [rsp+10h] [rbp-2D0h]
  __int64 v72; // [rsp+18h] [rbp-2C8h]
  char *v73; // [rsp+18h] [rbp-2C8h]
  __int64 v74; // [rsp+30h] [rbp-2B0h]
  _BYTE *v75; // [rsp+38h] [rbp-2A8h]
  unsigned __int64 v76; // [rsp+48h] [rbp-298h] BYREF
  __int64 v77; // [rsp+50h] [rbp-290h] BYREF
  char v78; // [rsp+60h] [rbp-280h]
  char v79[48]; // [rsp+70h] [rbp-270h] BYREF
  __int64 v80; // [rsp+A0h] [rbp-240h] BYREF
  void *s; // [rsp+A8h] [rbp-238h]
  __int64 v82; // [rsp+B0h] [rbp-230h]
  __int64 v83; // [rsp+B8h] [rbp-228h]
  __int64 *v84; // [rsp+C0h] [rbp-220h]
  __int64 v85; // [rsp+C8h] [rbp-218h]
  __int64 v86; // [rsp+D0h] [rbp-210h] BYREF
  char v87; // [rsp+D8h] [rbp-208h] BYREF
  _QWORD *v88; // [rsp+E0h] [rbp-200h]
  char *v89; // [rsp+E8h] [rbp-1F8h]
  _QWORD *v90; // [rsp+110h] [rbp-1D0h]
  char v91; // [rsp+130h] [rbp-1B0h]
  const char **v92; // [rsp+140h] [rbp-1A0h] BYREF
  __int64 v93; // [rsp+148h] [rbp-198h]
  char *v94; // [rsp+150h] [rbp-190h]
  __int64 v95; // [rsp+158h] [rbp-188h]
  __int64 v96; // [rsp+160h] [rbp-180h]
  __int64 v97; // [rsp+168h] [rbp-178h]
  __int64 v98; // [rsp+170h] [rbp-170h]
  unsigned __int64 v99; // [rsp+178h] [rbp-168h]
  __int64 v100; // [rsp+180h] [rbp-160h]
  __int64 v101; // [rsp+188h] [rbp-158h]
  _BYTE *v102; // [rsp+190h] [rbp-150h]
  __int64 v103; // [rsp+198h] [rbp-148h]
  _BYTE v104[256]; // [rsp+1A0h] [rbp-140h] BYREF
  __int64 v105; // [rsp+2A0h] [rbp-40h]

  LOWORD(v96) = 260;
  v92 = (const char **)&qword_502E468[8];
  sub_C7EAD0((__int64)&v77, &v92, 0, 1u, 0);
  if ( (v78 & 1) != 0 && (_DWORD)v77 )
    sub_C64ED0("Failed to open contextual profile file", 1u);
  v2 = v77;
  v77 = 0;
  v3 = *(char **)(v2 + 16);
  v4 = *(const char ***)(v2 + 8);
  v74 = v2;
  if ( (unsigned __int64)(v3 - (char *)v4) <= 3 )
  {
    v93 = v3 - (char *)v4;
    v5 = 0;
    v92 = v4;
  }
  else
  {
    v92 = *(const char ***)(v2 + 8);
    v5 = v3 - (char *)v4 - 4;
    v3 = (char *)v4 + 4;
    v93 = 4;
  }
  v95 = v5;
  v98 = 0x200000000LL;
  v102 = v104;
  v103 = 0x800000000LL;
  v94 = v3;
  v96 = 0;
  v97 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v105 = 0;
  sub_31568A0(&v86, &v92);
  v6 = v91 & 1;
  v91 = (2 * (v91 & 1)) | v91 & 0xFD;
  if ( v6 )
    sub_C64ED0("Failed to parse contextual profiles", 1u);
  v7 = &v87;
  v80 = 0;
  s = 0;
  v82 = 0;
  v83 = 0;
  v84 = &v86;
  v85 = 0;
  if ( v89 == &v87 )
  {
    v45 = 0;
    v46 = 0;
    goto LABEL_55;
  }
  v8 = (__int64)v89;
  v80 = 1;
LABEL_8:
  if ( HIDWORD(v82) )
  {
    v9 = v83;
    if ( (unsigned int)v83 <= 0x40 )
      goto LABEL_10;
    sub_C7D6A0((__int64)s, 8LL * (unsigned int)v83, 8);
    s = 0;
    v82 = 0;
    LODWORD(v83) = 0;
  }
  while ( 1 )
  {
    LODWORD(v85) = 0;
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(_QWORD *)(v8 + 32);
    v12 = *(_QWORD **)(v10 + 16);
    v13 = (_QWORD *)(v10 + 8);
    if ( !v12 )
      goto LABEL_22;
    do
    {
      while ( 1 )
      {
        v14 = v12[2];
        v15 = v12[3];
        if ( v11 <= v12[4] )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v15 )
          goto LABEL_18;
      }
      v13 = v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v14 );
LABEL_18:
    if ( (_QWORD *)(v10 + 8) == v13 )
      goto LABEL_22;
    if ( v11 < v13[4] )
      goto LABEL_22;
    v16 = *(_BYTE *)(v10 + 343) & 0xF8 | (unsigned __int64)(v13 + 4) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v16 )
      goto LABEL_22;
    v17 = *(_QWORD *)(v16 + 24);
    if ( *(_QWORD *)(v16 + 32) - v17 != 8 )
      goto LABEL_22;
    v28 = *(const void **)(*(_QWORD *)v17 + 24LL);
    v29 = *(_QWORD *)(*(_QWORD *)v17 + 32LL);
    v30 = sub_C92610();
    v31 = sub_C92740(a1 + 40, v28, v29, v30);
    v32 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v31);
    i = *v32;
    if ( *v32 )
    {
      if ( i != -8 )
        goto LABEL_39;
      --*(_DWORD *)(a1 + 56);
    }
    v69 = v32;
    v71 = v31;
    v62 = sub_C7D670(v29 + 41, 8);
    v63 = v71;
    v64 = v69;
    v65 = v62;
    if ( v29 )
    {
      v68 = v62;
      memcpy((void *)(v62 + 40), v28, v29);
      v63 = v71;
      v64 = v69;
      v65 = v68;
    }
    *(_BYTE *)(v65 + v29 + 40) = 0;
    *(_QWORD *)v65 = v29;
    *(_QWORD *)(v65 + 8) = 0;
    *(_QWORD *)(v65 + 16) = 0;
    *(_QWORD *)(v65 + 24) = 0;
    *(_DWORD *)(v65 + 32) = 0;
    *v64 = v65;
    ++*(_DWORD *)(a1 + 52);
    v66 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 40), v63));
    for ( i = *v66; !i; ++v66 )
LABEL_99:
      i = v66[1];
    if ( i == -8 )
      goto LABEL_99;
LABEL_39:
    v72 = i;
    sub_25D53E0(v8 + 40, (__int64)&v80);
    if ( &v84[(unsigned int)v85] != v84 )
      break;
LABEL_22:
    v8 = sub_220EF30(v8);
    if ( (char *)v8 == v7 )
      goto LABEL_52;
LABEL_23:
    ++v80;
    if ( !(_DWORD)v82 )
      goto LABEL_8;
    v18 = 4 * v82;
    v9 = v83;
    if ( (unsigned int)(4 * v82) < 0x40 )
      v18 = 64;
    if ( v18 >= (unsigned int)v83 )
    {
LABEL_10:
      if ( 8LL * v9 )
        memset(s, 255, 8LL * v9);
      v82 = 0;
      continue;
    }
    v19 = (char *)s;
    v20 = 8LL * (unsigned int)v83;
    if ( (_DWORD)v82 == 1 )
    {
      v25 = 1024;
      v24 = 128;
      goto LABEL_32;
    }
    _BitScanReverse(&v21, v82 - 1);
    v22 = 1 << (33 - (v21 ^ 0x1F));
    if ( v22 < 64 )
      v22 = 64;
    if ( v22 == (_DWORD)v83 )
    {
      v82 = 0;
      v67 = (char *)s + v20;
      do
      {
        if ( v19 )
          *(_QWORD *)v19 = -1;
        v19 += 8;
      }
      while ( v67 != v19 );
    }
    else
    {
      v23 = (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
          | (4 * v22 / 3u + 1)
          | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)
          | (((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
            | (4 * v22 / 3u + 1)
            | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4);
      v24 = ((v23 >> 8) | v23 | (((v23 >> 8) | v23) >> 16)) + 1;
      v25 = 8 * (((v23 >> 8) | v23 | (((v23 >> 8) | v23) >> 16)) + 1);
LABEL_32:
      sub_C7D6A0((__int64)s, v20, 8);
      LODWORD(v83) = v24;
      v26 = (_QWORD *)sub_C7D670(v25, 8);
      v82 = 0;
      s = v26;
      for ( j = &v26[(unsigned int)v83]; j != v26; ++v26 )
      {
        if ( v26 )
          *v26 = -1;
      }
    }
  }
  v34 = v72;
  v73 = v7;
  v35 = (unsigned __int64 *)&v84[(unsigned int)v85];
  v70 = v8;
  v36 = a1;
  v37 = (unsigned __int64 *)v84;
  v38 = v34;
  do
  {
    v39 = *(_QWORD *)(v36 + 24);
    v40 = *v37;
    v41 = *(_QWORD **)(v39 + 16);
    v42 = (_QWORD *)(v39 + 8);
    if ( v41 )
    {
      do
      {
        while ( 1 )
        {
          v43 = v41[2];
          v44 = v41[3];
          if ( v40 <= v41[4] )
            break;
          v41 = (_QWORD *)v41[3];
          if ( !v44 )
            goto LABEL_46;
        }
        v42 = v41;
        v41 = (_QWORD *)v41[2];
      }
      while ( v43 );
LABEL_46:
      if ( v42 != (_QWORD *)(v39 + 8) && v40 >= v42[4] )
      {
        v76 = *(unsigned __int8 *)(v39 + 343) | (unsigned __int64)(v42 + 4) & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v76 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          sub_25CF280((__int64)v79, v38 + 8, &v76);
      }
    }
    ++v37;
  }
  while ( v35 != v37 );
  a1 = v36;
  v7 = v73;
  v8 = sub_220EF30(v70);
  if ( (char *)v8 != v73 )
    goto LABEL_23;
LABEL_52:
  if ( v84 != &v86 )
    _libc_free((unsigned __int64)v84);
  v45 = s;
  v46 = 8LL * (unsigned int)v83;
LABEL_55:
  sub_C7D6A0((__int64)v45, v46, 8);
  if ( (v91 & 2) != 0 )
    sub_25CE1D0(&v86, v46);
  if ( (v91 & 1) != 0 )
  {
    if ( v86 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v86 + 8LL))(v86);
  }
  else
  {
    sub_25CD400(v90);
    sub_25CD8F0(v88);
  }
  v47 = 32LL * (unsigned int)v103;
  v75 = v102;
  v48 = &v102[v47];
  if ( v102 != &v102[v47] )
  {
    v49 = &v102[v47];
    do
    {
      v50 = *((_QWORD *)v49 - 3);
      v51 = *((_QWORD *)v49 - 2);
      v49 -= 32;
      v52 = v50;
      if ( v51 != v50 )
      {
        do
        {
          while ( 1 )
          {
            v53 = *(volatile signed __int32 **)(v52 + 8);
            if ( v53 )
            {
              if ( &_pthread_key_create )
              {
                v54 = _InterlockedExchangeAdd(v53 + 2, 0xFFFFFFFF);
              }
              else
              {
                v54 = *((_DWORD *)v53 + 2);
                *((_DWORD *)v53 + 2) = v54 - 1;
              }
              if ( v54 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v53 + 16LL))(v53);
                if ( &_pthread_key_create )
                {
                  v55 = _InterlockedExchangeAdd(v53 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v55 = *((_DWORD *)v53 + 3);
                  *((_DWORD *)v53 + 3) = v55 - 1;
                }
                if ( v55 == 1 )
                  break;
              }
            }
            v52 += 16LL;
            if ( v51 == v52 )
              goto LABEL_72;
          }
          v52 += 16LL;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v53 + 24LL))(v53);
        }
        while ( v51 != v52 );
LABEL_72:
        v50 = *((_QWORD *)v49 + 1);
      }
      if ( v50 )
      {
        v46 = *((_QWORD *)v49 + 3) - v50;
        j_j___libc_free_0(v50);
      }
    }
    while ( v75 != v49 );
    v48 = v102;
  }
  if ( v48 != v104 )
    _libc_free((unsigned __int64)v48);
  v56 = v100;
  v57 = v99;
  if ( v100 != v99 )
  {
    do
    {
      while ( 1 )
      {
        v58 = *(volatile signed __int32 **)(v57 + 8);
        if ( v58 )
        {
          if ( &_pthread_key_create )
          {
            v59 = _InterlockedExchangeAdd(v58 + 2, 0xFFFFFFFF);
          }
          else
          {
            v59 = *((_DWORD *)v58 + 2);
            *((_DWORD *)v58 + 2) = v59 - 1;
          }
          if ( v59 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v58 + 16LL))(v58);
            if ( &_pthread_key_create )
            {
              v60 = _InterlockedExchangeAdd(v58 + 3, 0xFFFFFFFF);
            }
            else
            {
              v60 = *((_DWORD *)v58 + 3);
              *((_DWORD *)v58 + 3) = v60 - 1;
            }
            if ( v60 == 1 )
              break;
          }
        }
        v57 += 16LL;
        if ( v56 == v57 )
          goto LABEL_90;
      }
      v57 += 16LL;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v58 + 24LL))(v58);
    }
    while ( v56 != v57 );
LABEL_90:
    v57 = v99;
  }
  if ( v57 )
  {
    v46 = v101 - v57;
    j_j___libc_free_0(v57);
  }
  result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v74 + 8LL))(v74, v46);
  if ( (v78 & 1) == 0 && v77 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v77 + 8LL))(v77);
  return result;
}
