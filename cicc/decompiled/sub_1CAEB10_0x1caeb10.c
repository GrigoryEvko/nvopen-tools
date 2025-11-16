// Function: sub_1CAEB10
// Address: 0x1caeb10
//
__int64 __fastcall sub_1CAEB10(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // r8
  __int64 *v6; // r9
  __int64 v7; // r15
  __int64 v8; // rdx
  char v9; // al
  unsigned int v10; // edx
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  _BYTE *v13; // rdx
  unsigned int v14; // eax
  __int64 *v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 **v18; // rbx
  __int64 *v19; // r14
  _QWORD *v20; // rdi
  __int64 v21; // r12
  _QWORD *v22; // rax
  _BYTE *v23; // rax
  int v24; // r11d
  __int64 *v25; // r10
  __int64 *v26; // rdi
  _QWORD *v27; // rax
  __int64 i; // rbx
  unsigned int v29; // eax
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 *v32; // rbx
  __int64 *v33; // r12
  __int64 v34; // rdi
  unsigned int v36; // edx
  __int64 v37; // rdi
  int v38; // r11d
  int v39; // r11d
  unsigned int v40; // edx
  const void *v41; // rdi
  int v42; // r11d
  __int64 *v43; // r10
  _BYTE *v44; // rdi
  unsigned int v45; // eax
  int v46; // r11d
  int v47; // r11d
  unsigned int v48; // eax
  __int64 *v49; // rbx
  __int64 v51; // [rsp+10h] [rbp-F0h]
  __int64 v52; // [rsp+20h] [rbp-E0h]
  __int64 v53; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+38h] [rbp-C8h]
  __int64 v55; // [rsp+40h] [rbp-C0h]
  __int64 v56; // [rsp+48h] [rbp-B8h]
  _BYTE *v57; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+58h] [rbp-A8h]
  _BYTE v59[32]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+80h] [rbp-80h] BYREF
  __int64 v61; // [rsp+88h] [rbp-78h]
  _QWORD *v62; // [rsp+90h] [rbp-70h]
  __int64 v63; // [rsp+98h] [rbp-68h]
  __int64 v64; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v65; // [rsp+A8h] [rbp-58h]
  _QWORD *v66; // [rsp+B0h] [rbp-50h]
  __int64 v67; // [rsp+B8h] [rbp-48h]
  __int64 v68; // [rsp+C0h] [rbp-40h]
  __int64 *v69; // [rsp+C8h] [rbp-38h]

  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v61 = 8;
  v60 = sub_22077B0(64);
  v2 = (__int64 *)(v60 + 24);
  v3 = sub_22077B0(512);
  v7 = *(_QWORD *)(a2 + 32);
  v65 = v60 + 24;
  v8 = v3 + 512;
  *(_QWORD *)(v60 + 24) = v3;
  v63 = v3;
  v67 = v3;
  v62 = (_QWORD *)v3;
  v66 = (_QWORD *)v3;
  v64 = v3 + 512;
  v69 = v2;
  v68 = v3 + 512;
  v51 = a2 + 24;
  if ( v7 == a2 + 24 )
    goto LABEL_63;
  do
  {
    while ( 1 )
    {
      if ( !v7 )
      {
        v57 = 0;
        BUG();
      }
      v57 = (_BYTE *)(v7 - 56);
      v52 = *(_QWORD *)(v7 + 24);
      if ( v52 != v7 + 16 )
        break;
LABEL_14:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v51 == v7 )
        goto LABEL_35;
    }
    while ( 1 )
    {
      if ( !v52 )
        BUG();
      v17 = *(_QWORD *)(v52 + 24);
      if ( v17 != v52 + 16 )
        break;
LABEL_25:
      v52 = *(_QWORD *)(v52 + 8);
      if ( v7 + 16 == v52 )
        goto LABEL_14;
    }
    while ( 1 )
    {
      if ( !v17 )
        BUG();
      if ( *(_BYTE *)(v17 - 8) == 56 )
      {
        v8 = 1LL - (*(_DWORD *)(v17 - 4) & 0xFFFFFFF);
        v18 = (__int64 **)(v17 - 24 + 24 * v8);
        if ( v18 != (__int64 **)(v17 - 24) )
          break;
      }
LABEL_24:
      v17 = *(_QWORD *)(v17 + 8);
      if ( v52 + 16 == v17 )
        goto LABEL_25;
    }
    while ( 1 )
    {
      v19 = *v18;
      if ( !sub_1642F90(**v18, 64) )
        goto LABEL_23;
      v9 = *((_BYTE *)v19 + 16);
      if ( v9 != 13 )
        break;
      v10 = *((_DWORD *)v19 + 8);
      v11 = v19[3];
      v12 = 1LL << ((unsigned __int8)v10 - 1);
      if ( v10 > 0x40 )
      {
        v4 = (v10 - 1) >> 6;
        v8 = *(_QWORD *)v11;
        if ( (*(_QWORD *)(v11 + 8 * v4) & v12) != 0 )
          goto LABEL_7;
      }
      else
      {
        v4 = 64 - v10;
        if ( (v12 & v11) != 0 )
        {
          v8 = (__int64)(v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
LABEL_7:
          if ( v8 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_8;
          goto LABEL_23;
        }
        v8 = (__int64)(v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
      }
      if ( v8 >= 0x80000000LL )
        goto LABEL_8;
LABEL_23:
      v18 += 3;
      if ( v18 == (__int64 **)(v17 - 24) )
        goto LABEL_24;
    }
    if ( v9 == 62 && sub_1CAD1F0((__int64)v19) )
      goto LABEL_23;
LABEL_8:
    if ( !(_DWORD)v56 )
    {
      ++v53;
LABEL_105:
      sub_1CAE960((__int64)&v53, 2 * v56);
      if ( (_DWORD)v56 )
      {
        v44 = v57;
        LODWORD(v6) = v54;
        v4 = (unsigned int)(v55 + 1);
        v45 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v43 = (__int64 *)(v54 + 8LL * v45);
        v13 = (_BYTE *)*v43;
        if ( v57 == (_BYTE *)*v43 )
          goto LABEL_97;
        v46 = 1;
        v5 = 0;
        while ( v13 != (_BYTE *)-8LL )
        {
          if ( v13 == (_BYTE *)-16LL && !v5 )
            v5 = v43;
          v45 = (v56 - 1) & (v46 + v45);
          v43 = (__int64 *)(v54 + 8LL * v45);
          v13 = (_BYTE *)*v43;
          if ( v57 == (_BYTE *)*v43 )
            goto LABEL_97;
          ++v46;
        }
LABEL_109:
        v13 = v44;
        if ( v5 )
          v43 = v5;
        goto LABEL_97;
      }
LABEL_136:
      LODWORD(v55) = v55 + 1;
      BUG();
    }
    v13 = v57;
    LODWORD(v6) = v56 - 1;
    LODWORD(v5) = v54;
    v14 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
    v15 = (__int64 *)(v54 + 8LL * v14);
    v4 = *v15;
    if ( v57 == (_BYTE *)*v15 )
      goto LABEL_10;
    v42 = 1;
    v43 = 0;
    while ( v4 != -8 )
    {
      if ( v43 || v4 != -16 )
        v15 = v43;
      v14 = (unsigned int)v6 & (v42 + v14);
      v49 = (__int64 *)(v54 + 8LL * v14);
      v4 = *v49;
      if ( v57 == (_BYTE *)*v49 )
        goto LABEL_10;
      ++v42;
      v43 = v15;
      v15 = (__int64 *)(v54 + 8LL * v14);
    }
    if ( !v43 )
      v43 = v15;
    ++v53;
    v4 = (unsigned int)(v55 + 1);
    if ( 4 * (int)v4 >= (unsigned int)(3 * v56) )
      goto LABEL_105;
    if ( (int)v56 - HIDWORD(v55) - (int)v4 <= (unsigned int)v56 >> 3 )
    {
      sub_1CAE960((__int64)&v53, v56);
      if ( (_DWORD)v56 )
      {
        v44 = v57;
        v5 = 0;
        LODWORD(v6) = v54;
        v47 = 1;
        v4 = (unsigned int)(v55 + 1);
        v48 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v43 = (__int64 *)(v54 + 8LL * v48);
        v13 = (_BYTE *)*v43;
        if ( v57 == (_BYTE *)*v43 )
          goto LABEL_97;
        while ( v13 != (_BYTE *)-8LL )
        {
          if ( !v5 && v13 == (_BYTE *)-16LL )
            v5 = v43;
          v48 = (v56 - 1) & (v47 + v48);
          v43 = (__int64 *)(v54 + 8LL * v48);
          v13 = (_BYTE *)*v43;
          if ( v57 == (_BYTE *)*v43 )
            goto LABEL_97;
          ++v47;
        }
        goto LABEL_109;
      }
      goto LABEL_136;
    }
LABEL_97:
    LODWORD(v55) = v4;
    if ( *v43 != -8 )
      --HIDWORD(v55);
    *v43 = (__int64)v13;
LABEL_10:
    v16 = v66;
    v8 = v68 - 8;
    if ( v66 != (_QWORD *)(v68 - 8) )
    {
      if ( v66 )
      {
        v8 = (__int64)v57;
        *v66 = v57;
        v16 = v66;
      }
      v66 = v16 + 1;
      goto LABEL_14;
    }
    sub_1C6FE60(&v60, &v57);
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v51 != v7 );
LABEL_35:
  v20 = v66;
  if ( v66 != v62 )
  {
    while ( 1 )
    {
      if ( (_QWORD *)v67 == v20 )
      {
        v21 = *(_QWORD *)(*(v69 - 1) + 504);
        j_j___libc_free_0(v20, 512);
        v8 = *--v69 + 512;
        v67 = *v69;
        v68 = v8;
        v66 = (_QWORD *)(v67 + 504);
      }
      else
      {
        v21 = *(v20 - 1);
        v66 = v20 - 1;
      }
LABEL_38:
      v21 = *(_QWORD *)(v21 + 8);
      if ( v21 )
        break;
LABEL_53:
      v20 = v66;
      if ( v66 == v62 )
        goto LABEL_54;
    }
    while ( 2 )
    {
      v22 = sub_1648700(v21);
      if ( *((_BYTE *)v22 + 16) <= 0x17u )
        goto LABEL_38;
      v23 = *(_BYTE **)(v22[5] + 56LL);
      v57 = v23;
      if ( (_DWORD)v56 )
      {
        LODWORD(v6) = v56 - 1;
        LODWORD(v5) = v54;
        v24 = 1;
        v25 = 0;
        v8 = ((_DWORD)v56 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v26 = (__int64 *)(v54 + 8 * v8);
        v4 = *v26;
        if ( v23 == (_BYTE *)*v26 )
          goto LABEL_38;
        while ( v4 != -8 )
        {
          if ( v4 != -16 || v25 )
            v26 = v25;
          v8 = (unsigned int)v6 & (v24 + (_DWORD)v8);
          v4 = *(_QWORD *)(v54 + 8LL * (unsigned int)v8);
          if ( v23 == (_BYTE *)v4 )
            goto LABEL_38;
          ++v24;
          v25 = v26;
          v26 = (__int64 *)(v54 + 8LL * (unsigned int)v8);
        }
        if ( !v25 )
          v25 = v26;
        ++v53;
        v4 = (unsigned int)(v55 + 1);
        if ( 4 * (int)v4 < (unsigned int)(3 * v56) )
        {
          if ( (int)v56 - HIDWORD(v55) - (int)v4 <= (unsigned int)v56 >> 3 )
          {
            sub_1CAE960((__int64)&v53, v56);
            if ( !(_DWORD)v56 )
            {
LABEL_139:
              LODWORD(v55) = v55 + 1;
              BUG();
            }
            v23 = v57;
            LODWORD(v5) = v54;
            v6 = 0;
            v39 = 1;
            v40 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
            v25 = (__int64 *)(v54 + 8LL * v40);
            v41 = (const void *)*v25;
            v4 = (unsigned int)(v55 + 1);
            if ( (_BYTE *)*v25 != v57 )
            {
              while ( v41 != (const void *)-8LL )
              {
                if ( !v6 && v41 == (const void *)-16LL )
                  v6 = v25;
                v40 = (v56 - 1) & (v39 + v40);
                v25 = (__int64 *)(v54 + 8LL * v40);
                v41 = (const void *)*v25;
                if ( v57 == (_BYTE *)*v25 )
                  goto LABEL_47;
                ++v39;
              }
              goto LABEL_80;
            }
          }
          goto LABEL_47;
        }
      }
      else
      {
        ++v53;
      }
      sub_1CAE960((__int64)&v53, 2 * v56);
      if ( !(_DWORD)v56 )
        goto LABEL_139;
      v23 = v57;
      LODWORD(v5) = v54;
      v36 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v25 = (__int64 *)(v54 + 8LL * v36);
      v37 = *v25;
      v4 = (unsigned int)(v55 + 1);
      if ( (_BYTE *)*v25 != v57 )
      {
        v38 = 1;
        v6 = 0;
        while ( v37 != -8 )
        {
          if ( !v6 && v37 == -16 )
            v6 = v25;
          v36 = (v56 - 1) & (v38 + v36);
          v25 = (__int64 *)(v54 + 8LL * v36);
          v37 = *v25;
          if ( v57 == (_BYTE *)*v25 )
            goto LABEL_47;
          ++v38;
        }
LABEL_80:
        if ( v6 )
          v25 = v6;
      }
LABEL_47:
      LODWORD(v55) = v4;
      if ( *v25 != -8 )
        --HIDWORD(v55);
      *v25 = (__int64)v23;
      v27 = v66;
      v8 = v68 - 8;
      if ( v66 == (_QWORD *)(v68 - 8) )
      {
        sub_1C6FE60(&v60, &v57);
        goto LABEL_38;
      }
      if ( v66 )
      {
        v8 = (__int64)v57;
        *v66 = v57;
        v27 = v66;
      }
      v66 = v27 + 1;
      v21 = *(_QWORD *)(v21 + 8);
      if ( !v21 )
        goto LABEL_53;
      continue;
    }
  }
LABEL_54:
  for ( i = *(_QWORD *)(a2 + 32); v51 != i; i = *(_QWORD *)(i + 8) )
  {
LABEL_58:
    v30 = 0;
    if ( i )
      v30 = i - 56;
    if ( (_DWORD)v56 )
    {
      v8 = (unsigned int)(v56 - 1);
      LODWORD(v5) = 1;
      v29 = v8 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v4 = *(_QWORD *)(v54 + 8LL * v29);
      if ( v30 == v4 )
        goto LABEL_57;
      while ( v4 != -8 )
      {
        LODWORD(v6) = (_DWORD)v5 + 1;
        v29 = v8 & ((_DWORD)v5 + v29);
        v4 = *(_QWORD *)(v54 + 8LL * v29);
        if ( v30 == v4 )
          goto LABEL_57;
        LODWORD(v5) = (_DWORD)v5 + 1;
      }
    }
    v58 = 0x2000000000LL;
    v57 = v59;
    sub_1CAD270(v30, (__int64)&v57, v8, v4, (int)v5, (int)v6);
    sub_1632440(a2, v57, (unsigned int)v58);
    if ( v57 == v59 )
    {
LABEL_57:
      i = *(_QWORD *)(i + 8);
      if ( v51 == i )
        break;
      goto LABEL_58;
    }
    _libc_free((unsigned __int64)v57);
  }
LABEL_63:
  v31 = v60;
  if ( v60 )
  {
    v32 = (__int64 *)v65;
    v33 = v69 + 1;
    if ( (unsigned __int64)(v69 + 1) > v65 )
    {
      do
      {
        v34 = *v32++;
        j_j___libc_free_0(v34, 512);
      }
      while ( v33 > v32 );
      v31 = v60;
    }
    j_j___libc_free_0(v31, 8 * v61);
  }
  return j___libc_free_0(v54);
}
