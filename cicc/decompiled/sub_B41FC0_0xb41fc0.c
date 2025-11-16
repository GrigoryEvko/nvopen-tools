// Function: sub_B41FC0
// Address: 0xb41fc0
//
__int64 __fastcall sub_B41FC0(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v4; // r12
  int v5; // ebx
  int v6; // eax
  int v7; // r9d
  size_t v8; // r10
  __int64 *v9; // r8
  size_t v10; // r11
  unsigned int v11; // r15d
  __int64 v12; // r14
  int v13; // ecx
  int v14; // r15d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rcx
  unsigned int v19; // edx
  __int64 v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r12
  const void *v25; // rsi
  int v26; // eax
  char *v27; // rsi
  char *v28; // rdx
  void **p_s1; // rsi
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 v34; // rax
  _QWORD *v35; // r12
  _QWORD *v36; // r15
  _QWORD *v37; // r12
  _QWORD *v38; // rbx
  __int64 *v39; // r14
  __int64 v40; // rdx
  __int64 *v41; // r13
  __int64 *v42; // rbx
  __int64 v43; // rax
  __int64 *v44; // r12
  __int64 *v45; // r15
  __int64 *v46; // r12
  __int64 *v47; // rbx
  const void *v48; // rdi
  int v49; // eax
  _QWORD *v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 v53; // rbx
  __int64 *v54; // r14
  __int64 *v55; // rsi
  unsigned __int64 v56; // rax
  __int64 *v57; // rbx
  __int64 *v58; // rdi
  __int64 *v59; // rdi
  size_t v60; // [rsp+8h] [rbp-718h]
  size_t v61; // [rsp+10h] [rbp-710h]
  __int64 *v62; // [rsp+18h] [rbp-708h]
  __int64 *v63; // [rsp+20h] [rbp-700h]
  int v64; // [rsp+28h] [rbp-6F8h]
  unsigned __int16 v65; // [rsp+2Eh] [rbp-6F2h]
  void *s2; // [rsp+30h] [rbp-6F0h]
  __int64 *v67; // [rsp+38h] [rbp-6E8h]
  __int64 v68; // [rsp+40h] [rbp-6E0h]
  char *v69; // [rsp+48h] [rbp-6D8h]
  __int64 v70; // [rsp+50h] [rbp-6D0h]
  __int64 *v71; // [rsp+58h] [rbp-6C8h]
  char *v72; // [rsp+60h] [rbp-6C0h] BYREF
  size_t v73; // [rsp+68h] [rbp-6B8h]
  _QWORD v74[2]; // [rsp+70h] [rbp-6B0h] BYREF
  unsigned __int16 v75; // [rsp+80h] [rbp-6A0h]
  __int64 *v76; // [rsp+90h] [rbp-690h] BYREF
  __int64 v77; // [rsp+98h] [rbp-688h]
  _BYTE v78[768]; // [rsp+A0h] [rbp-680h] BYREF
  __int64 v79; // [rsp+3A0h] [rbp-380h]
  void *s1; // [rsp+3B0h] [rbp-370h] BYREF
  size_t n; // [rsp+3B8h] [rbp-368h]
  _QWORD v82[2]; // [rsp+3C0h] [rbp-360h] BYREF
  unsigned __int16 v83; // [rsp+3D0h] [rbp-350h]
  _BYTE *v84; // [rsp+3D8h] [rbp-348h] BYREF
  __int64 v85; // [rsp+3E0h] [rbp-340h]
  _BYTE v86[768]; // [rsp+3E8h] [rbp-338h] BYREF
  __int64 v87; // [rsp+6E8h] [rbp-38h]

  v69 = (char *)v74;
  v72 = (char *)v74;
  sub_B3B150((__int64 *)&v72, a2, (__int64)&a2[a3]);
  v75 = 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 31;
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 16);
    v16 = *(unsigned int *)(a1 + 24);
    v4 = v15;
    v5 = v16 - 1;
    if ( !(_DWORD)v16 )
      goto LABEL_8;
  }
  v71 = v82;
  s1 = v82;
  n = 0;
  LOBYTE(v82[0]) = 0;
  v83 = 0;
  v6 = sub_B3B940(v72, &v72[v73]);
  v7 = v75;
  LODWORD(v70) = 1;
  v8 = v73;
  v9 = (__int64 *)s1;
  v10 = n;
  v11 = v5 & v6;
  LOBYTE(v67) = HIBYTE(v75);
  LOBYTE(v68) = v75;
  s2 = v72;
  v65 = v83;
  while ( 1 )
  {
    v12 = v4 + 48LL * v11;
    v13 = *(unsigned __int16 *)(v12 + 32);
    if ( (_WORD)v13 == (_WORD)v7 )
    {
      if ( !(_BYTE)v68 )
        goto LABEL_10;
      if ( (_BYTE)v67 )
        goto LABEL_10;
      if ( *(_QWORD *)(v12 + 8) == v8 )
      {
        v60 = v10;
        LODWORD(v61) = v7;
        v64 = v13;
        if ( !v8 )
          goto LABEL_10;
        v48 = *(const void **)v12;
        v62 = v9;
        v63 = (__int64 *)v8;
        v49 = memcmp(v48, s2, v8);
        v8 = (size_t)v63;
        v9 = v62;
        LOWORD(v13) = v64;
        v7 = v61;
        v10 = v60;
        if ( !v49 )
        {
LABEL_10:
          if ( v9 != v71 )
            j_j___libc_free_0(v9, v82[0] + 1LL);
          if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
          {
            v15 = *(_QWORD *)(a1 + 16);
            goto LABEL_14;
          }
          goto LABEL_30;
        }
      }
    }
    if ( (_WORD)v13 == v65 )
    {
      if ( !*(_BYTE *)(v12 + 32) )
        break;
      if ( *(_BYTE *)(v12 + 33) )
        break;
      if ( v10 == *(_QWORD *)(v12 + 8) )
      {
        v61 = v8;
        v64 = v7;
        if ( !v10 )
          break;
        v25 = *(const void **)v12;
        v62 = (__int64 *)v10;
        v63 = v9;
        v26 = memcmp(v9, v25, v10);
        v9 = v63;
        v10 = (size_t)v62;
        v7 = v64;
        v8 = v61;
        if ( !v26 )
          break;
      }
    }
    v14 = v70 + v11;
    LODWORD(v70) = v70 + 1;
    v11 = v5 & v14;
  }
  if ( v9 != v71 )
    j_j___libc_free_0(v9, v82[0] + 1LL);
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v12 = a1 + 1552;
LABEL_30:
    v15 = a1 + 16;
    v17 = 1536;
    goto LABEL_15;
  }
  v15 = *(_QWORD *)(a1 + 16);
  v16 = *(unsigned int *)(a1 + 24);
LABEL_8:
  v12 = v15 + 48 * v16;
LABEL_14:
  v17 = 48LL * *(unsigned int *)(a1 + 24);
LABEL_15:
  v18 = *(_QWORD *)(a1 + 28192);
  v19 = *(_DWORD *)(a1 + 1560);
  v20 = *(_QWORD *)(a1 + 1552);
  if ( v12 == v17 + v15 )
  {
    v21 = v19;
    v22 = v20 + 832LL * v19;
  }
  else
  {
    v21 = v19;
    v22 = v20 + 832LL * *(unsigned int *)(v12 + 40);
    if ( v22 != v20 + 832LL * v19 )
    {
      v23 = v22 + 40;
      *(_QWORD *)(a1 + 28192) = v18 + 1;
      *(_QWORD *)(v22 + 824) = v18;
      goto LABEL_18;
    }
  }
  if ( v21 <= 0x20 || (unsigned __int64)(v18 - *(_QWORD *)(a1 + 28200)) <= 0x60 )
  {
    v71 = v82;
  }
  else
  {
    v50 = v82;
    v71 = v82;
    s1 = v82;
    n = 0x2000000000LL;
    if ( v22 == v20 )
    {
      v54 = v71;
    }
    else
    {
      v51 = 0;
      while ( 1 )
      {
        v50[v51] = v20;
        v20 += 832;
        v51 = (unsigned int)(n + 1);
        LODWORD(n) = n + 1;
        if ( v20 == v22 )
          break;
        if ( v51 + 1 > (unsigned __int64)HIDWORD(n) )
        {
          sub_C8D5F0(&s1, v71, v51 + 1, 8);
          v51 = (unsigned int)n;
        }
        v50 = s1;
      }
      v52 = (__int64 *)s1;
      v53 = 8 * v51;
      v54 = (__int64 *)((char *)s1 + 8 * v51);
      if ( s1 != v54 )
      {
        v55 = (__int64 *)((char *)s1 + 8 * v51);
        _BitScanReverse64(&v56, v53 >> 3);
        sub_B41CA0((__int64 *)s1, v55, 2LL * (int)(63 - (v56 ^ 0x3F)), a1);
        if ( (unsigned __int64)v53 <= 0x80 )
        {
          sub_B40450(v52, v54, a1);
        }
        else
        {
          v57 = v52 + 16;
          sub_B40450(v52, v52 + 16, a1);
          if ( v54 != v52 + 16 )
          {
            do
            {
              v58 = v57++;
              sub_B403E0(v58, a1);
            }
            while ( v54 != v57 );
          }
        }
        v54 = (__int64 *)s1;
      }
    }
    v76 = *(__int64 **)(sub_B3FBB0(a1, v54[31]) + 784);
    sub_B40500(a1, (char *)&v76);
    v18 = *(_QWORD *)(a1 + 28192);
    v59 = (__int64 *)s1;
    *(_QWORD *)(a1 + 28200) = v18;
    if ( v59 != v71 )
    {
      _libc_free(v59, &v76);
      v18 = *(_QWORD *)(a1 + 28192);
    }
  }
  v27 = v72;
  v67 = (__int64 *)v78;
  v76 = (__int64 *)v78;
  v28 = &v72[v73];
  *(_QWORD *)(a1 + 28192) = v18 + 1;
  v77 = 0x400000000LL;
  s1 = v71;
  v79 = v18;
  sub_B3AE60((__int64 *)&s1, v27, (__int64)v28);
  v85 = 0x400000000LL;
  v83 = v75;
  s2 = v86;
  v84 = v86;
  if ( (_DWORD)v77 )
    sub_B3E940((__int64 *)&v84, (__int64)&v76);
  p_s1 = &s1;
  v87 = v79;
  v68 = sub_B40C20(a1, (__int64)&s1, (__int64)&v84);
  v70 = (__int64)v84;
  v30 = (__int64)&v84[192 * (unsigned int)v85];
  if ( v84 != (_BYTE *)v30 )
  {
    do
    {
      v31 = *(unsigned int *)(v30 - 120);
      v32 = *(_QWORD *)(v30 - 128);
      v30 -= 192;
      v33 = v32 + 56 * v31;
      if ( v32 != v33 )
      {
        do
        {
          v34 = *(unsigned int *)(v33 - 40);
          v35 = *(_QWORD **)(v33 - 48);
          v33 -= 56;
          v34 *= 32;
          v36 = (_QWORD *)((char *)v35 + v34);
          if ( v35 != (_QWORD *)((char *)v35 + v34) )
          {
            do
            {
              v36 -= 4;
              if ( (_QWORD *)*v36 != v36 + 2 )
              {
                p_s1 = (void **)(v36[2] + 1LL);
                j_j___libc_free_0(*v36, p_s1);
              }
            }
            while ( v35 != v36 );
            v35 = *(_QWORD **)(v33 + 8);
          }
          if ( v35 != (_QWORD *)(v33 + 24) )
            _libc_free(v35, p_s1);
        }
        while ( v32 != v33 );
        v32 = *(_QWORD *)(v30 + 64);
      }
      if ( v32 != v30 + 80 )
        _libc_free(v32, p_s1);
      v37 = *(_QWORD **)(v30 + 16);
      v38 = &v37[4 * *(unsigned int *)(v30 + 24)];
      if ( v37 != v38 )
      {
        do
        {
          v38 -= 4;
          if ( (_QWORD *)*v38 != v38 + 2 )
          {
            p_s1 = (void **)(v38[2] + 1LL);
            j_j___libc_free_0(*v38, p_s1);
          }
        }
        while ( v37 != v38 );
        v37 = *(_QWORD **)(v30 + 16);
      }
      if ( v37 != (_QWORD *)(v30 + 32) )
        _libc_free(v37, p_s1);
    }
    while ( v70 != v30 );
    v30 = (__int64)v84;
  }
  if ( (void *)v30 != s2 )
    _libc_free(v30, p_s1);
  if ( s1 != v71 )
  {
    p_s1 = (void **)(v82[0] + 1LL);
    j_j___libc_free_0(s1, v82[0] + 1LL);
  }
  v71 = v76;
  v39 = &v76[24 * (unsigned int)v77];
  if ( v76 != v39 )
  {
    do
    {
      v40 = *((unsigned int *)v39 - 30);
      v41 = (__int64 *)*(v39 - 16);
      v39 -= 24;
      v42 = &v41[7 * v40];
      if ( v41 != v42 )
      {
        do
        {
          v43 = *((unsigned int *)v42 - 10);
          v44 = (__int64 *)*(v42 - 6);
          v42 -= 7;
          v43 *= 32;
          v45 = (__int64 *)((char *)v44 + v43);
          if ( v44 != (__int64 *)((char *)v44 + v43) )
          {
            do
            {
              v45 -= 4;
              if ( (__int64 *)*v45 != v45 + 2 )
              {
                p_s1 = (void **)(v45[2] + 1);
                j_j___libc_free_0(*v45, p_s1);
              }
            }
            while ( v44 != v45 );
            v44 = (__int64 *)v42[1];
          }
          if ( v44 != v42 + 3 )
            _libc_free(v44, p_s1);
        }
        while ( v41 != v42 );
        v41 = (__int64 *)v39[8];
      }
      if ( v41 != v39 + 10 )
        _libc_free(v41, p_s1);
      v46 = (__int64 *)v39[2];
      v47 = &v46[4 * *((unsigned int *)v39 + 6)];
      if ( v46 != v47 )
      {
        do
        {
          v47 -= 4;
          if ( (__int64 *)*v47 != v47 + 2 )
          {
            p_s1 = (void **)(v47[2] + 1);
            j_j___libc_free_0(*v47, p_s1);
          }
        }
        while ( v46 != v47 );
        v46 = (__int64 *)v39[2];
      }
      if ( v46 != v39 + 4 )
        _libc_free(v46, p_s1);
    }
    while ( v71 != v39 );
    v39 = v76;
  }
  if ( v39 != v67 )
    _libc_free(v39, p_s1);
  v23 = v68 + 40;
LABEL_18:
  if ( v72 != v69 )
    j_j___libc_free_0(v72, v74[0] + 1LL);
  return v23;
}
