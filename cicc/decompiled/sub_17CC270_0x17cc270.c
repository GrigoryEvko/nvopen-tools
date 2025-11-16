// Function: sub_17CC270
// Address: 0x17cc270
//
__int64 __fastcall sub_17CC270(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *i; // rdx
  __int64 v16; // rax
  _BYTE *v17; // rdi
  __int64 v18; // rdx
  size_t v19; // rcx
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 k; // r12
  __int64 v33; // rsi
  char v34; // al
  unsigned int v36; // ecx
  _QWORD *v37; // rdi
  unsigned int v38; // eax
  int v39; // eax
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rax
  int v42; // ebx
  __int64 v43; // r12
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _QWORD *j; // rdx
  char *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  char *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  size_t v53; // rdx
  _QWORD *v54; // rax
  __int64 v55; // [rsp+0h] [rbp-C0h]
  __int64 v57; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v58; // [rsp+1Fh] [rbp-A1h]
  __int64 v59; // [rsp+20h] [rbp-A0h]
  __int64 v60; // [rsp+30h] [rbp-90h] BYREF
  __int16 v61; // [rsp+40h] [rbp-80h]
  _QWORD *v62; // [rsp+50h] [rbp-70h] BYREF
  size_t n; // [rsp+58h] [rbp-68h]
  _QWORD src[12]; // [rsp+60h] [rbp-60h] BYREF

  v12 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 104) = a3;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  if ( !v12 )
  {
    if ( !*(_DWORD *)(a1 + 132) )
      goto LABEL_7;
    v13 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)v13 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 120));
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v36 = 4 * v12;
  v13 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)(4 * v12) < 0x40 )
    v36 = 64;
  if ( (unsigned int)v13 <= v36 )
  {
LABEL_4:
    v14 = *(_QWORD **)(a1 + 120);
    for ( i = &v14[4 * v13]; i != v14; v14 += 4 )
      *v14 = -8;
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_7;
  }
  v37 = *(_QWORD **)(a1 + 120);
  v38 = v12 - 1;
  if ( !v38 )
  {
    v43 = 4096;
    v42 = 128;
LABEL_52:
    j___libc_free_0(v37);
    *(_DWORD *)(a1 + 136) = v42;
    v44 = (_QWORD *)sub_22077B0(v43);
    v45 = *(unsigned int *)(a1 + 136);
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 120) = v44;
    for ( j = &v44[4 * v45]; j != v44; v44 += 4 )
    {
      if ( v44 )
        *v44 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v38, v38);
  v39 = 1 << (33 - (v38 ^ 0x1F));
  if ( v39 < 64 )
    v39 = 64;
  if ( (_DWORD)v13 != v39 )
  {
    v40 = (4 * v39 / 3u + 1) | ((unsigned __int64)(4 * v39 / 3u + 1) >> 1);
    v41 = ((v40 | (v40 >> 2)) >> 4) | v40 | (v40 >> 2) | ((((v40 | (v40 >> 2)) >> 4) | v40 | (v40 >> 2)) >> 8);
    v42 = (v41 | (v41 >> 16)) + 1;
    v43 = 32 * ((v41 | (v41 >> 16)) + 1);
    goto LABEL_52;
  }
  *(_QWORD *)(a1 + 128) = 0;
  v54 = &v37[4 * (unsigned int)v13];
  do
  {
    if ( v37 )
      *v37 = -8;
    v37 += 4;
  }
  while ( v54 != v37 );
LABEL_7:
  v16 = *(_QWORD *)(a1 + 144);
  if ( *(_QWORD *)(a1 + 152) != v16 )
    *(_QWORD *)(a1 + 152) = v16;
  sub_1695A80((char *)qword_4FA3D40[20], qword_4FA3D40[21], (_QWORD *)(a1 + 232), (_QWORD *)(a1 + 240));
  v61 = 260;
  v60 = a2 + 240;
  sub_16E1010((__int64)&v62, (__int64)&v60);
  v17 = *(_BYTE **)(a1 + 48);
  if ( v62 == src )
  {
    v53 = n;
    if ( n )
    {
      if ( n == 1 )
        *v17 = src[0];
      else
        memcpy(v17, src, n);
      v53 = n;
      v17 = *(_BYTE **)(a1 + 48);
    }
    *(_QWORD *)(a1 + 56) = v53;
    v17[v53] = 0;
    v17 = v62;
  }
  else
  {
    v18 = src[0];
    v19 = n;
    if ( v17 == (_BYTE *)(a1 + 64) )
    {
      *(_QWORD *)(a1 + 48) = v62;
      *(_QWORD *)(a1 + 56) = v19;
      *(_QWORD *)(a1 + 64) = v18;
    }
    else
    {
      v20 = *(_QWORD *)(a1 + 64);
      *(_QWORD *)(a1 + 48) = v62;
      *(_QWORD *)(a1 + 56) = v19;
      *(_QWORD *)(a1 + 64) = v18;
      if ( v17 )
      {
        v62 = v17;
        src[0] = v20;
        goto LABEL_13;
      }
    }
    v62 = src;
    v17 = src;
  }
LABEL_13:
  n = 0;
  *v17 = 0;
  v21 = v62;
  *(_QWORD *)(a1 + 80) = src[2];
  *(_QWORD *)(a1 + 88) = src[3];
  *(_QWORD *)(a1 + 96) = src[4];
  if ( v21 != src )
    j_j___libc_free_0(v21, src[0] + 1LL);
  v58 = sub_17C7710(a1);
  v55 = sub_16321C0(a2, (__int64)"__llvm_coverage_names", 21, 1);
  v22 = sub_15E0FD0(110);
  v24 = sub_16321A0(a2, (__int64)v22, v23);
  if ( !v24 || !*(_QWORD *)(v24 + 8) )
  {
    v47 = sub_15E0FD0(111);
    v49 = sub_16321A0(a2, (__int64)v47, v48);
    if ( !v49 || !*(_QWORD *)(v49 + 8) )
    {
      v50 = sub_15E0FD0(112);
      v52 = sub_16321A0(a2, (__int64)v50, v51);
      if ( !v52 || !*(_QWORD *)(v52 + 8) )
      {
        if ( !v55 )
          return v58;
        v57 = a2 + 24;
        v59 = *(_QWORD *)(a2 + 32);
        if ( v59 == a2 + 24 )
          goto LABEL_67;
        while ( 1 )
        {
LABEL_18:
          if ( !v59 )
            BUG();
          v27 = *(_QWORD *)(v59 + 24);
          v28 = 0;
          if ( v59 + 16 == v27 )
            goto LABEL_35;
          do
          {
            if ( !v27 )
              BUG();
            v29 = *(_QWORD *)(v27 + 24);
            v30 = v27 + 16;
            if ( v27 + 16 != v29 )
            {
              while ( 1 )
              {
                if ( !v29 )
                  BUG();
                if ( *(_BYTE *)(v29 - 8) != 78 )
                  goto LABEL_23;
                v31 = *(_QWORD *)(v29 - 48);
                if ( *(_BYTE *)(v31 + 16) )
                  goto LABEL_23;
                if ( (*(_BYTE *)(v31 + 33) & 0x20) != 0 && *(_DWORD *)(v31 + 36) == 112 )
                {
                  sub_17CBE70(a1, v29 - 24);
LABEL_23:
                  v29 = *(_QWORD *)(v29 + 8);
                  if ( v30 == v29 )
                    break;
                }
                else
                {
                  if ( v28 || (*(_BYTE *)(v31 + 33) & 0x20) == 0 || *(_DWORD *)(v31 + 36) != 110 )
                    goto LABEL_23;
                  v28 = v29 - 24;
                  v29 = *(_QWORD *)(v29 + 8);
                  if ( v30 == v29 )
                    break;
                }
              }
            }
            v27 = *(_QWORD *)(v27 + 8);
          }
          while ( v59 + 16 != v27 );
          if ( v28 )
            sub_17CA660(a1, v28);
LABEL_35:
          v59 = *(_QWORD *)(v59 + 8);
          if ( v57 == v59 )
          {
            for ( k = *(_QWORD *)(a2 + 32); v59 != k; v58 |= v34 )
            {
              v33 = k - 56;
              if ( !k )
                v33 = 0;
              v34 = sub_17CBD40((_QWORD *)a1, v33, a4, a5, a6, a7, v25, v26, a10, a11);
              k = *(_QWORD *)(k + 8);
            }
            goto LABEL_40;
          }
        }
      }
    }
  }
  v57 = a2 + 24;
  v59 = *(_QWORD *)(a2 + 32);
  if ( a2 + 24 != v59 )
    goto LABEL_18;
LABEL_40:
  if ( v55 )
  {
LABEL_67:
    sub_17C7310(a1, v55);
    goto LABEL_42;
  }
  if ( v58 )
  {
LABEL_42:
    sub_17C7CB0(a1);
    sub_17C7480(a1);
    sub_17C6BE0((_QWORD *)a1);
    sub_17C5B00((_QWORD *)a1);
    sub_17C6C10(a1);
    return 1;
  }
  return v58;
}
