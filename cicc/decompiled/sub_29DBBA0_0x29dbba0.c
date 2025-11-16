// Function: sub_29DBBA0
// Address: 0x29dbba0
//
__int64 *__fastcall sub_29DBBA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v8; // rax
  size_t v9; // r14
  const void *v10; // r15
  size_t v11; // r14
  __int64 v12; // rbx
  int v13; // eax
  int v14; // eax
  __int64 *v15; // rax
  __int64 v16; // r14
  void *v17; // rdx
  const char *v18; // r8
  size_t v19; // rbx
  __int64 v20; // r9
  unsigned __int64 v21; // rcx
  _BYTE *v22; // r13
  size_t v23; // r15
  __int64 *v24; // rax
  size_t v25; // r15
  __int64 v26; // r9
  __int64 v27; // rdi
  _BYTE *v28; // rsi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rdi
  const void *v33; // r15
  _BYTE *v34; // rdi
  __int64 *v35; // rax
  size_t v36; // r15
  __int64 *v37; // r14
  const char *v38; // rax
  size_t v39; // rdx
  __int64 v40; // r9
  const char *v41; // r8
  size_t v42; // rax
  _BYTE *v43; // rax
  __int64 v44; // rdx
  _BYTE *v45; // rsi
  _BYTE *v46; // rdi
  __int64 v47; // r8
  unsigned __int64 v48; // rdx
  _BYTE *v49; // rdi
  __int64 v50; // rbx
  const char *v51; // [rsp+8h] [rbp-298h]
  const char *v52; // [rsp+8h] [rbp-298h]
  size_t v53; // [rsp+10h] [rbp-290h]
  size_t v54; // [rsp+10h] [rbp-290h]
  size_t v55; // [rsp+10h] [rbp-290h]
  __int64 *src; // [rsp+18h] [rbp-288h]
  void *srca; // [rsp+18h] [rbp-288h]
  void *srcb; // [rsp+18h] [rbp-288h]
  size_t srcc; // [rsp+18h] [rbp-288h]
  _BYTE *v60; // [rsp+28h] [rbp-278h] BYREF
  __int64 *v61; // [rsp+30h] [rbp-270h] BYREF
  size_t n; // [rsp+38h] [rbp-268h]
  __int64 v63; // [rsp+40h] [rbp-260h] BYREF
  _BYTE dest[264]; // [rsp+48h] [rbp-258h] BYREF
  _BYTE *v65; // [rsp+150h] [rbp-150h] BYREF
  __int64 v66; // [rsp+158h] [rbp-148h]
  unsigned __int64 v67; // [rsp+160h] [rbp-140h] BYREF
  _BYTE v68[312]; // [rsp+168h] [rbp-138h] BYREF

  v8 = *(_QWORD **)(a3 + 40);
  if ( !(_BYTE)qword_50090A8 || (v9 = v8[26]) == 0 )
  {
    v10 = (const void *)v8[21];
    v11 = v8[22];
    v12 = *(_QWORD *)(a2 + 8);
    v13 = sub_C92610();
    v14 = sub_C92860((__int64 *)(v12 + 48), v10, v11, v13);
    if ( v14 == -1 )
      v15 = (__int64 *)(*(_QWORD *)(v12 + 48) + 8LL * *(unsigned int *)(v12 + 56));
    else
      v15 = (__int64 *)(*(_QWORD *)(v12 + 48) + 8LL * v14);
    v16 = *v15;
    v18 = sub_BD5D20(a3);
    v19 = (size_t)v17;
    v20 = (__int64)v17;
    v21 = *(unsigned int *)(v16 + 12) | ((unsigned __int64)*(unsigned int *)(v16 + 8) << 32);
    if ( !v21 )
    {
      BYTE4(v67) = 48;
      v22 = (char *)&v67 + 4;
      v61 = &v63;
LABEL_7:
      v23 = 1;
      LOBYTE(v63) = *v22;
      v24 = &v63;
      goto LABEL_8;
    }
    v22 = (char *)&v67 + 5;
    do
    {
      *--v22 = v21 % 0xA + 48;
      v30 = v21;
      v21 /= 0xAu;
    }
    while ( v30 > 9 );
    v23 = (char *)&v67 + 5 - v22;
    v61 = &v63;
    v60 = (_BYTE *)((char *)&v67 + 5 - v22);
    if ( (unsigned __int64)((char *)&v67 + 5 - v22) <= 0xF )
    {
      if ( v23 == 1 )
        goto LABEL_7;
      if ( !v23 )
      {
        v24 = &v63;
LABEL_8:
        n = v23;
        *((_BYTE *)v24 + v23) = 0;
        v65 = v68;
        v25 = n;
        src = v61;
        v66 = 0;
        v67 = 256;
        if ( v19 > 0x100 )
        {
          v52 = v18;
          sub_C8D290((__int64)&v65, v68, v19, 1u, (__int64)v18, v20);
          v18 = v52;
          v49 = &v65[v66];
        }
        else
        {
          if ( !v19 )
          {
LABEL_10:
            v26 = (__int64)&v65[v20];
            *(_DWORD *)v26 = 1986817070;
            *(_WORD *)(v26 + 4) = 11885;
            v27 = v66 + 6;
            v66 = v27;
            if ( v25 + v27 > v67 )
            {
              sub_C8D290((__int64)&v65, v68, v25 + v27, 1u, (__int64)v18, v26);
              v27 = v66;
            }
            v28 = v65;
            if ( v25 )
            {
              memcpy(&v65[v27], src, v25);
              v28 = v65;
              v27 = v66;
            }
            *a1 = (__int64)(a1 + 2);
            v66 = v25 + v27;
            sub_29DB790(a1, v28, (__int64)&v28[v25 + v27]);
            if ( v65 != v68 )
              _libc_free((unsigned __int64)v65);
            if ( v61 != &v63 )
              j_j___libc_free_0((unsigned __int64)v61);
            return a1;
          }
          v49 = v68;
        }
        memcpy(v49, v18, v19);
        v50 = v66 + v19;
        v66 = v50;
        v20 = v50;
        if ( v50 + 6 > v67 )
        {
          sub_C8D290((__int64)&v65, v68, v50 + 6, 1u, (__int64)v18, v50);
          v20 = v66;
        }
        goto LABEL_10;
      }
      v32 = &v63;
    }
    else
    {
      v53 = (size_t)v18;
      srca = v17;
      v31 = sub_22409D0((__int64)&v61, (unsigned __int64 *)&v60, 0);
      v20 = (__int64)srca;
      v18 = (const char *)v53;
      v61 = (__int64 *)v31;
      v32 = (__int64 *)v31;
      v63 = (__int64)v60;
    }
    v54 = (size_t)v18;
    srcb = (void *)v20;
    memcpy(v32, v22, v23);
    v23 = (size_t)v60;
    v24 = v61;
    v20 = (__int64)srcb;
    v18 = (const char *)v54;
    goto LABEL_8;
  }
  v33 = (const void *)v8[25];
  n = 0;
  v61 = (__int64 *)dest;
  v34 = dest;
  v63 = 256;
  if ( v9 > 0x100 )
  {
    sub_C8D290((__int64)&v61, dest, v9, 1u, (__int64)&v61, a6);
    v34 = (char *)v61 + n;
  }
  memcpy(v34, v33, v9);
  v35 = v61;
  v36 = v9 + n;
  v37 = (__int64 *)((char *)v61 + v9 + n);
  n = v36;
  if ( v61 != v37 )
  {
    do
    {
      if ( (unsigned __int8)((*(_BYTE *)v35 & 0xDF) - 65) > 0x19u && (unsigned __int8)(*(_BYTE *)v35 - 48) > 9u )
        *(_BYTE *)v35 = 95;
      v35 = (__int64 *)((char *)v35 + 1);
    }
    while ( v37 != v35 );
    v37 = v61;
    v36 = n;
  }
  v38 = sub_BD5D20(a3);
  v65 = v68;
  v66 = 0;
  v41 = v38;
  v42 = v39;
  v67 = 256;
  if ( v39 > 0x100 )
  {
    v51 = v41;
    v55 = v39;
    sub_C8D290((__int64)&v65, v68, v39, 1u, (__int64)v41, v40);
    v39 = v55;
    v41 = v51;
    v46 = &v65[v66];
    goto LABEL_44;
  }
  if ( v39 )
  {
    v46 = v68;
LABEL_44:
    srcc = v39;
    memcpy(v46, v41, v39);
    v42 = srcc + v66;
    v48 = srcc + v66 + 6;
    v66 += srcc;
    if ( v48 > v67 )
    {
      sub_C8D290((__int64)&v65, v68, v48, 1u, v47, v40);
      v42 = v66;
    }
  }
  v43 = &v65[v42];
  *((_WORD *)v43 + 2) = 11885;
  *(_DWORD *)v43 = 1986817070;
  v44 = v66 + 6;
  v66 = v44;
  if ( v44 + v36 > v67 )
  {
    sub_C8D290((__int64)&v65, v68, v44 + v36, 1u, v44 + v36, v40);
    v44 = v66;
  }
  v45 = v65;
  if ( v36 )
  {
    memcpy(&v65[v44], v37, v36);
    v45 = v65;
    v44 = v66;
  }
  *a1 = (__int64)(a1 + 2);
  v66 = v36 + v44;
  sub_29DB790(a1, v45, (__int64)&v45[v36 + v44]);
  if ( v65 != v68 )
    _libc_free((unsigned __int64)v65);
  if ( v61 != (__int64 *)dest )
    _libc_free((unsigned __int64)v61);
  return a1;
}
