// Function: sub_3174140
// Address: 0x3174140
//
__int64 __fastcall sub_3174140(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  char *v8; // r14
  __int64 v9; // rax
  __int64 i; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  char *v19; // r14
  __int64 j; // r15
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r8
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // r8
  unsigned __int64 v38; // rdx
  int v39; // r14d
  void *v40; // r8
  size_t v41; // r15
  __int64 *v42; // r13
  __int64 v43; // r14
  __int64 *v44; // r15
  unsigned __int64 v45; // rax
  __int64 *v46; // r12
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r14
  __int64 v50; // rbx
  _QWORD *v51; // rdi
  __int64 *v53; // r14
  __int64 v54; // rbx
  __int64 *k; // r15
  __int64 v56; // rdx
  __int64 *v57; // r13
  __int64 *v58; // rdi
  __int64 v59; // [rsp+8h] [rbp-4B8h]
  __int64 v60; // [rsp+10h] [rbp-4B0h]
  __int64 v62; // [rsp+20h] [rbp-4A0h]
  char *dest; // [rsp+30h] [rbp-490h]
  char *desta; // [rsp+30h] [rbp-490h]
  __int64 *v65; // [rsp+38h] [rbp-488h]
  __int64 v66; // [rsp+38h] [rbp-488h]
  void *v67; // [rsp+38h] [rbp-488h]
  _BYTE *v68; // [rsp+40h] [rbp-480h] BYREF
  __int64 v69; // [rsp+48h] [rbp-478h]
  _BYTE v70[256]; // [rsp+50h] [rbp-470h] BYREF
  __int64 v71; // [rsp+150h] [rbp-370h] BYREF
  __int64 v72; // [rsp+158h] [rbp-368h]
  __int64 v73; // [rsp+160h] [rbp-360h]
  __int64 v74; // [rsp+168h] [rbp-358h]
  void *src; // [rsp+170h] [rbp-350h]
  __int64 v76; // [rsp+178h] [rbp-348h]
  _BYTE v77[256]; // [rsp+180h] [rbp-340h] BYREF
  __int64 *v78; // [rsp+280h] [rbp-240h] BYREF
  __int64 v79; // [rsp+288h] [rbp-238h]
  _BYTE v80[560]; // [rsp+290h] [rbp-230h] BYREF

  v7 = a2;
  v8 = *(char **)(a3 + 144);
  src = v77;
  v76 = 0x2000000000LL;
  v69 = 0x2000000000LL;
  v9 = *(unsigned int *)(a3 + 152);
  v68 = v70;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  for ( dest = &v8[40 * v9]; dest != v8; v8 += 40 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)v8 + 16LL); i; i = *(_QWORD *)(i + 8) )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(i + 24);
        v12 = *(_QWORD *)(a2 + 40);
        v78 = (__int64 *)v11;
        if ( *(_QWORD *)(v11 + 40) == v12
          && !(unsigned __int8)sub_B19DB0(a1, a2, v11)
          && (unsigned __int8)sub_3173AD0((__int64)&v71, (__int64 *)&v78, v13, v14, v15, a6) )
        {
          break;
        }
        i = *(_QWORD *)(i + 8);
        if ( !i )
          goto LABEL_11;
      }
      v16 = (unsigned int)v69;
      v17 = (__int64)v78;
      v18 = (unsigned int)v69 + 1LL;
      if ( v18 > HIDWORD(v69) )
      {
        v59 = (__int64)v78;
        sub_C8D5F0((__int64)&v68, v70, v18, 8u, (__int64)v78, a6);
        v16 = (unsigned int)v69;
        v17 = v59;
      }
      *(_QWORD *)&v68[8 * v16] = v17;
      LODWORD(v69) = v69 + 1;
    }
LABEL_11:
    ;
  }
  v19 = *a4;
  desta = &(*a4)[48 * *((unsigned int *)a4 + 2)];
  if ( *a4 != desta )
  {
    do
    {
      for ( j = *(_QWORD *)(*(_QWORD *)v19 + 16LL); j; j = *(_QWORD *)(j + 8) )
      {
        while ( 1 )
        {
          v21 = *(_QWORD *)(j + 24);
          v22 = *(_QWORD *)(a2 + 40);
          v78 = (__int64 *)v21;
          if ( *(_QWORD *)(v21 + 40) == v22
            && !(unsigned __int8)sub_B19DB0(a1, a2, v21)
            && (unsigned __int8)sub_3173AD0((__int64)&v71, (__int64 *)&v78, v23, v24, v25, a6) )
          {
            break;
          }
          j = *(_QWORD *)(j + 8);
          if ( !j )
            goto LABEL_22;
        }
        v26 = (unsigned int)v69;
        v27 = (__int64)v78;
        v28 = (unsigned int)v69 + 1LL;
        if ( v28 > HIDWORD(v69) )
        {
          v62 = (__int64)v78;
          sub_C8D5F0((__int64)&v68, v70, v28, 8u, (__int64)v78, a6);
          v26 = (unsigned int)v69;
          v27 = v62;
        }
        *(_QWORD *)&v68[8 * v26] = v27;
        LODWORD(v69) = v69 + 1;
      }
LABEL_22:
      v19 += 48;
    }
    while ( desta != v19 );
  }
  v29 = v69;
  while ( v29 )
  {
    v30 = v29--;
    v31 = *(_QWORD *)&v68[8 * v30 - 8];
    LODWORD(v69) = v29;
    v32 = *(_QWORD *)(v31 + 16);
    if ( v32 )
    {
      do
      {
        while ( 1 )
        {
          v78 = *(__int64 **)(v32 + 24);
          if ( !(unsigned __int8)sub_B19DB0(a1, a2, (__int64)v78) )
          {
            if ( (unsigned __int8)sub_3173AD0((__int64)&v71, (__int64 *)&v78, v33, v34, v35, a6) )
              break;
          }
          v32 = *(_QWORD *)(v32 + 8);
          if ( !v32 )
            goto LABEL_32;
        }
        v36 = (unsigned int)v69;
        v37 = (__int64)v78;
        v38 = (unsigned int)v69 + 1LL;
        if ( v38 > HIDWORD(v69) )
        {
          v66 = (__int64)v78;
          sub_C8D5F0((__int64)&v68, v70, v38, 8u, (__int64)v78, a6);
          v36 = (unsigned int)v69;
          v37 = v66;
        }
        *(_QWORD *)&v68[8 * v36] = v37;
        LODWORD(v69) = v69 + 1;
        v32 = *(_QWORD *)(v32 + 8);
      }
      while ( v32 );
LABEL_32:
      v29 = v69;
    }
  }
  v39 = v76;
  v40 = src;
  v78 = (__int64 *)v80;
  v41 = 8LL * (unsigned int)v76;
  v79 = 0x4000000000LL;
  if ( (unsigned int)v76 > 0x40uLL )
  {
    v67 = src;
    sub_C8D5F0((__int64)&v78, v80, (unsigned int)v76, 8u, (__int64)src, a6);
    v40 = v67;
    v58 = &v78[(unsigned int)v79];
  }
  else
  {
    v42 = (__int64 *)v80;
    if ( !v41 )
      goto LABEL_36;
    v58 = (__int64 *)v80;
  }
  memcpy(v58, v40, v41);
  v42 = v78;
  LODWORD(v41) = v79;
LABEL_36:
  LODWORD(v79) = v41 + v39;
  v43 = (unsigned int)(v41 + v39);
  v65 = &v42[v43];
  v44 = &v42[v43];
  if ( &v42[v43] == v42 )
  {
    v47 = *(_QWORD *)(a2 + 32);
    if ( v47 == *(_QWORD *)(a2 + 40) + 48LL )
      goto LABEL_46;
    v46 = v42;
    if ( !v47 )
      goto LABEL_46;
    goto LABEL_41;
  }
  _BitScanReverse64(&v45, (v43 * 8) >> 3);
  sub_316FC70(v42, &v42[v43], 2LL * (int)(63 - (v45 ^ 0x3F)), a1);
  if ( (unsigned __int64)v43 > 16 )
  {
    v53 = v42 + 16;
    sub_316FA50(v42, v42 + 16, a1);
    if ( v44 != v42 + 16 )
    {
      do
      {
        v54 = *v53;
        for ( k = v53; ; k[1] = *k )
        {
          v56 = *(k - 1);
          v57 = k--;
          if ( !(unsigned __int8)sub_B19DB0(a1, v54, v56) )
            break;
        }
        *v57 = v54;
        ++v53;
      }
      while ( v65 != v53 );
      v7 = a2;
    }
  }
  else
  {
    sub_316FA50(v42, v65, a1);
  }
  v42 = v78;
  v46 = &v78[(unsigned int)v79];
  v47 = *(_QWORD *)(v7 + 32);
  if ( v47 != *(_QWORD *)(v7 + 40) + 48LL && v47 )
  {
LABEL_41:
    v48 = v47 - 24;
    goto LABEL_42;
  }
  v48 = 0;
LABEL_42:
  if ( v46 != v42 )
  {
    v49 = v60;
    v50 = v48 + 24;
    do
    {
      v51 = (_QWORD *)*v42;
      LOWORD(v49) = 0;
      ++v42;
      sub_B444E0(v51, v50, v49);
    }
    while ( v46 != v42 );
    v42 = v78;
  }
LABEL_46:
  if ( v42 != (__int64 *)v80 )
    _libc_free((unsigned __int64)v42);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  if ( src != v77 )
    _libc_free((unsigned __int64)src);
  return sub_C7D6A0(v72, 8LL * (unsigned int)v74, 8);
}
