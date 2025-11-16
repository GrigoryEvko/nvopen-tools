// Function: sub_19A1660
// Address: 0x19a1660
//
__int64 __fastcall sub_19A1660(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rax
  unsigned __int64 *v13; // r15
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  unsigned __int64 *v16; // rax
  unsigned __int64 *v17; // rsi
  char v18; // al
  int v19; // r9d
  __int64 v20; // rdi
  unsigned int v22; // esi
  int v23; // eax
  int v24; // eax
  int v25; // r8d
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rcx
  int v29; // r8d
  int v30; // r9d
  unsigned int v31; // eax
  __int64 v32; // r12
  __int64 *v33; // r15
  __int64 *v34; // r12
  __int64 *v35; // r10
  __int64 *v36; // r9
  __int64 v37; // rsi
  __int64 *v38; // rdi
  unsigned int v39; // r8d
  __int64 *v40; // rax
  __int64 *v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 *v44; // rax
  __int64 *v45; // r13
  __int64 *i; // rbx
  __int64 v47; // rsi
  unsigned __int64 *v48; // rsi
  unsigned __int64 *src; // [rsp+0h] [rbp-B0h]
  char *v52; // [rsp+20h] [rbp-90h] BYREF
  __int64 v53; // [rsp+28h] [rbp-88h]
  _BYTE v54[32]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v55[12]; // [rsp+50h] [rbp-60h] BYREF

  v8 = *(unsigned int *)(a2 + 752);
  if ( (_DWORD)v8 && *(_BYTE *)(a2 + 729) )
    return 0;
  v9 = *(unsigned int *)(a4 + 40);
  v52 = v54;
  v53 = 0x400000000LL;
  if ( !(_DWORD)v9 )
  {
    if ( !*(_QWORD *)(a4 + 80) )
      goto LABEL_12;
    goto LABEL_5;
  }
  sub_19930D0((__int64)&v52, a4 + 32, v9, v8, a5, a6);
  if ( *(_QWORD *)(a4 + 80) )
LABEL_5:
    sub_1458920((__int64)&v52, (_QWORD *)(a4 + 80));
  v10 = 8LL * (unsigned int)v53;
  v11 = (unsigned __int64 *)&v52[v10];
  if ( v52 != &v52[v10] )
  {
    src = (unsigned __int64 *)v52;
    _BitScanReverse64(&v12, v10 >> 3);
    sub_1993A10(v52, (unsigned __int64 *)&v52[v10], 2LL * (int)(63 - (v12 ^ 0x3F)));
    if ( (unsigned __int64)v10 <= 0x80 )
    {
      sub_1992E50(src, v11);
    }
    else
    {
      v13 = src + 16;
      sub_1992E50(src, src + 16);
      if ( v11 != src + 16 )
      {
        do
        {
          while ( 1 )
          {
            v14 = *v13;
            v15 = *(v13 - 1);
            v16 = v13 - 1;
            if ( *v13 < v15 )
              break;
            v48 = v13++;
            *v48 = v14;
            if ( v11 == v13 )
              goto LABEL_12;
          }
          do
          {
            v16[1] = v15;
            v17 = v16;
            v15 = *--v16;
          }
          while ( v14 < v15 );
          ++v13;
          *v17 = v14;
        }
        while ( v11 != v13 );
      }
    }
  }
LABEL_12:
  v18 = sub_19A0320(a2, (__int64)&v52, v55);
  v20 = v55[0];
  if ( v18 )
  {
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
    return 0;
  }
  v22 = *(_DWORD *)(a2 + 24);
  v23 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v24 = v23 + 1;
  v25 = 2 * v22;
  if ( 4 * v24 >= 3 * v22 )
  {
    v22 *= 2;
    goto LABEL_57;
  }
  v26 = v22 - *(_DWORD *)(a2 + 20) - v24;
  v27 = v22 >> 3;
  if ( (unsigned int)v26 <= (unsigned int)v27 )
  {
LABEL_57:
    sub_19A0530(a2, v22);
    sub_19A0320(a2, (__int64)&v52, v55);
    v20 = v55[0];
    v24 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v24;
  v55[2] = -1;
  if ( *(_DWORD *)(v20 + 8) != 1 || (v26 = -1, **(_QWORD **)v20 != -1) )
    --*(_DWORD *)(a2 + 20);
  sub_19930D0(v20, (__int64)&v52, v26, v27, v25, v19);
  v31 = *(_DWORD *)(a2 + 752);
  if ( v31 >= *(_DWORD *)(a2 + 756) )
  {
    sub_1995E60(a2 + 744, 0);
    v31 = *(_DWORD *)(a2 + 752);
  }
  v32 = *(_QWORD *)(a2 + 744) + 96LL * v31;
  if ( v32 )
  {
    *(_QWORD *)v32 = *(_QWORD *)a4;
    *(_QWORD *)(v32 + 8) = *(_QWORD *)(a4 + 8);
    *(_BYTE *)(v32 + 16) = *(_BYTE *)(a4 + 16);
    *(_QWORD *)(v32 + 24) = *(_QWORD *)(a4 + 24);
    *(_QWORD *)(v32 + 32) = v32 + 48;
    *(_QWORD *)(v32 + 40) = 0x400000000LL;
    if ( *(_DWORD *)(a4 + 40) )
      sub_19930D0(v32 + 32, a4 + 32, v31, v28, v29, v30);
    *(_QWORD *)(v32 + 80) = *(_QWORD *)(a4 + 80);
    *(_QWORD *)(v32 + 88) = *(_QWORD *)(a4 + 88);
    v31 = *(_DWORD *)(a2 + 752);
  }
  *(_DWORD *)(a2 + 752) = v31 + 1;
  v33 = *(__int64 **)(a4 + 32);
  v34 = &v33[*(unsigned int *)(a4 + 40)];
  if ( v33 != v34 )
  {
    v35 = *(__int64 **)(a2 + 1928);
    v36 = *(__int64 **)(a2 + 1920);
    do
    {
LABEL_31:
      v37 = *v33;
      if ( v36 != v35 )
        goto LABEL_29;
      v38 = &v36[*(unsigned int *)(a2 + 1940)];
      v39 = *(_DWORD *)(a2 + 1940);
      if ( v38 != v36 )
      {
        v40 = v36;
        v41 = 0;
        while ( v37 != *v40 )
        {
          if ( *v40 == -2 )
            v41 = v40;
          if ( v38 == ++v40 )
          {
            if ( !v41 )
              goto LABEL_52;
            ++v33;
            *v41 = v37;
            v35 = *(__int64 **)(a2 + 1928);
            --*(_DWORD *)(a2 + 1944);
            v36 = *(__int64 **)(a2 + 1920);
            ++*(_QWORD *)(a2 + 1912);
            if ( v34 != v33 )
              goto LABEL_31;
            goto LABEL_40;
          }
        }
        goto LABEL_30;
      }
LABEL_52:
      if ( v39 < *(_DWORD *)(a2 + 1936) )
      {
        *(_DWORD *)(a2 + 1940) = v39 + 1;
        *v38 = v37;
        v36 = *(__int64 **)(a2 + 1920);
        ++*(_QWORD *)(a2 + 1912);
        v35 = *(__int64 **)(a2 + 1928);
      }
      else
      {
LABEL_29:
        sub_16CCBA0(a2 + 1912, v37);
        v35 = *(__int64 **)(a2 + 1928);
        v36 = *(__int64 **)(a2 + 1920);
      }
LABEL_30:
      ++v33;
    }
    while ( v34 != v33 );
  }
LABEL_40:
  v42 = *(_QWORD *)(a4 + 80);
  if ( v42 )
    sub_199AF80((__int64)v55, a2 + 1912, v42);
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  v43 = *(_QWORD *)(a4 + 80);
  if ( v43 )
    sub_1998430(a1 + 32128, v43, a3);
  v44 = *(__int64 **)(a4 + 32);
  v45 = &v44[*(unsigned int *)(a4 + 40)];
  for ( i = v44; v45 != i; ++i )
  {
    v47 = *i;
    sub_1998430(a1 + 32128, v47, a3);
  }
  return 1;
}
