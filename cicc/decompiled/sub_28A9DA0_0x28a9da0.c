// Function: sub_28A9DA0
// Address: 0x28a9da0
//
__int64 __fastcall sub_28A9DA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // r12
  __int64 v9; // rsi
  unsigned __int64 v10; // r9
  unsigned __int64 v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  __int64 i; // rbx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // r15
  __int64 v24; // rdx
  char **v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r12
  unsigned __int64 v30; // rcx
  __int64 result; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  const void *v35; // r15
  __int64 v36; // r8
  int v37; // ebx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rcx
  __int64 v42; // rbx
  __int64 v43; // r15
  __int64 v44; // rdx
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rax
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdi
  char *v50; // rax
  unsigned __int64 v51; // rsi
  char *v52; // rbx
  unsigned __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r14
  char *v57; // rbx
  char *v58; // rbx
  __int64 v59; // [rsp+0h] [rbp-120h]
  char *v60; // [rsp+8h] [rbp-118h]
  char *v61; // [rsp+18h] [rbp-108h]
  __int64 v62; // [rsp+18h] [rbp-108h]
  __int16 v63; // [rsp+26h] [rbp-FAh]
  __int64 v66; // [rsp+38h] [rbp-E8h]
  __int128 v67; // [rsp+40h] [rbp-E0h] BYREF
  __int128 v68; // [rsp+50h] [rbp-D0h]
  _BYTE *v69; // [rsp+60h] [rbp-C0h]
  __int64 v70; // [rsp+68h] [rbp-B8h]
  _BYTE v71[176]; // [rsp+70h] [rbp-B0h] BYREF

  v6 = a2 + a3;
  v8 = a1;
  v9 = *(unsigned int *)(a1 + 8);
  v66 = v6;
  v10 = *(_QWORD *)a1;
  v63 = a5;
  v11 = v10;
  v12 = 176 * v9;
  v13 = v9;
  v14 = v10 + 176 * v9;
  for ( i = 0x2E8BA2E8BA2E8BA3LL * ((176 * v9) >> 4); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      a5 = i >> 1;
      v16 = v11 + 176 * (i >> 1);
      if ( a2 <= *(_QWORD *)(v16 + 8) )
        break;
      v11 = v16 + 176;
      i = i - a5 - 1;
      if ( i <= 0 )
        goto LABEL_5;
    }
  }
LABEL_5:
  if ( v14 == v11 )
  {
    v51 = v9 + 1;
    v52 = (char *)&v67;
    v69 = v71;
    v70 = 0x1000000000LL;
    v53 = *(unsigned int *)(v8 + 12);
    v67 = 0;
    v68 = 0;
    if ( v51 > v53 )
    {
      if ( v10 > (unsigned __int64)&v67 || v11 <= (unsigned __int64)&v67 )
      {
        sub_28A9920(v8, v51, v14, v13, a5, v10);
        v10 = *(_QWORD *)v8;
      }
      else
      {
        v58 = (char *)&v67 - v10;
        sub_28A9920(v8, v51, v14, v13, a5, v10);
        v10 = *(_QWORD *)v8;
        v52 = &v58[*(_QWORD *)v8];
      }
      v13 = *(unsigned int *)(v8 + 8);
      a5 = 176 * v13;
      v11 = v10 + 176 * v13;
    }
    if ( v11 )
    {
      *(_QWORD *)v11 = *(_QWORD *)v52;
      *(_QWORD *)(v11 + 8) = *((_QWORD *)v52 + 1);
      *(_QWORD *)(v11 + 16) = *((_QWORD *)v52 + 2);
      *(_WORD *)(v11 + 24) = *((_WORD *)v52 + 12);
      *(_QWORD *)(v11 + 32) = v11 + 48;
      *(_QWORD *)(v11 + 40) = 0x1000000000LL;
      v54 = *((unsigned int *)v52 + 10);
      if ( (_DWORD)v54 )
        sub_28A9600(v11 + 32, (char **)v52 + 4, v54, v13, a5, v10);
      v10 = *(_QWORD *)v8;
      LODWORD(v13) = *(_DWORD *)(v8 + 8);
    }
    v55 = (unsigned int)(v13 + 1);
    *(_DWORD *)(v8 + 8) = v55;
    v11 = v10 + 176 * v55 - 176;
    v29 = v11 + 32;
    goto LABEL_18;
  }
  if ( *(_QWORD *)v11 > v66 )
  {
    v17 = v9 + 1;
    v69 = v71;
    v70 = 0x1000000000LL;
    v18 = *(unsigned int *)(v8 + 12);
    v60 = (char *)&v67;
    v61 = (char *)&v67;
    v67 = 0;
    v68 = 0;
    if ( v17 > v18 )
    {
      v56 = v11 - v10;
      if ( v10 > (unsigned __int64)&v67 || v14 <= (unsigned __int64)&v67 )
      {
        sub_28A9920(v8, v17, v14, v13, a5, v10);
        v10 = *(_QWORD *)v8;
        v61 = (char *)&v67;
        v11 = *(_QWORD *)v8 + v56;
        v13 = *(unsigned int *)(v8 + 8);
        v12 = 176 * v13;
        v14 = *(_QWORD *)v8 + 176 * v13;
      }
      else
      {
        v57 = (char *)&v67 - v10;
        sub_28A9920(v8, v17, v14, v13, a5, v10);
        v10 = *(_QWORD *)v8;
        v11 = *(_QWORD *)v8 + v56;
        v13 = *(unsigned int *)(v8 + 8);
        v61 = &v57[*(_QWORD *)v8];
        v12 = 176 * v13;
        v60 = v61;
        v14 = *(_QWORD *)v8 + 176 * v13;
      }
    }
    v19 = v10 + v12 - 176;
    if ( v14 )
    {
      *(_QWORD *)v14 = *(_QWORD *)v19;
      *(_QWORD *)(v14 + 8) = *(_QWORD *)(v19 + 8);
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v19 + 16);
      *(_WORD *)(v14 + 24) = *(_WORD *)(v19 + 24);
      *(_QWORD *)(v14 + 32) = v14 + 48;
      *(_QWORD *)(v14 + 40) = 0x1000000000LL;
      if ( *(_DWORD *)(v19 + 40) )
        sub_28A9600(v14 + 32, (char **)(v19 + 32), v14, v13, a5, v10);
      v10 = *(_QWORD *)v8;
      v13 = *(unsigned int *)(v8 + 8);
      v14 = *(_QWORD *)v8 + 176 * v13;
      v19 = v14 - 176;
    }
    v20 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(v19 - v11) >> 4);
    if ( (__int64)(v19 - v11) > 0 )
    {
      v59 = v8;
      v21 = v19 - 144;
      v22 = v14 - 144;
      v23 = v20;
      do
      {
        v24 = *(_QWORD *)(v21 - 32);
        v25 = (char **)v21;
        v26 = v22;
        v21 -= 176;
        v22 -= 176;
        *(_QWORD *)(v22 + 144) = v24;
        *(_QWORD *)(v22 + 152) = *(_QWORD *)(v21 + 152);
        *(_QWORD *)(v22 + 160) = *(_QWORD *)(v21 + 160);
        v27 = *(unsigned __int16 *)(v21 + 168);
        *(_WORD *)(v22 + 168) = v27;
        sub_28A9600(v26, v25, v27, v13, a5, v10);
        --v23;
      }
      while ( v23 );
      v8 = v59;
      LODWORD(v13) = *(_DWORD *)(v59 + 8);
      v10 = *(_QWORD *)v59;
    }
    v28 = (unsigned int)(v13 + 1);
    *(_DWORD *)(v8 + 8) = v28;
    if ( v11 <= (unsigned __int64)v60 )
    {
      v10 += 176 * v28;
      v50 = v61 + 176;
      if ( v10 <= (unsigned __int64)v60 )
        v50 = v61;
      v61 = v50;
    }
    v29 = v11 + 32;
    *(_QWORD *)v11 = *(_QWORD *)v61;
    *(_QWORD *)(v11 + 8) = *((_QWORD *)v61 + 1);
    *(_QWORD *)(v11 + 16) = *((_QWORD *)v61 + 2);
    *(_WORD *)(v11 + 24) = *((_WORD *)v61 + 12);
    sub_28A9600(v11 + 32, (char **)v61 + 4, v14, (__int64)v61, a5, v10);
LABEL_18:
    if ( v69 != v71 )
      _libc_free((unsigned __int64)v69);
    v30 = *(unsigned int *)(v11 + 44);
    *(_QWORD *)v11 = a2;
    *(_QWORD *)(v11 + 8) = v66;
    *(_QWORD *)(v11 + 16) = a4;
    *(_WORD *)(v11 + 24) = v63;
    result = *(unsigned int *)(v11 + 40);
    if ( result + 1 > v30 )
    {
      sub_C8D5F0(v29, (const void *)(v11 + 48), result + 1, 8u, a5, v10);
      result = *(unsigned int *)(v11 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8 * result) = a6;
    ++*(_DWORD *)(v11 + 40);
    return result;
  }
  v62 = v11 + 32;
  v32 = *(unsigned int *)(v11 + 40);
  if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 44) )
  {
    sub_C8D5F0(v62, (const void *)(v11 + 48), v32 + 1, 8u, a5, v10);
    v32 = *(unsigned int *)(v11 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v11 + 32) + 8 * v32) = a6;
  ++*(_DWORD *)(v11 + 40);
  if ( *(_QWORD *)v11 > a2 )
  {
    *(_QWORD *)v11 = a2;
    *(_QWORD *)(v11 + 16) = a4;
    *(_WORD *)(v11 + 24) = v63;
    result = v66;
    if ( v66 <= *(_QWORD *)(v11 + 8) )
      return result;
  }
  else
  {
    result = v66;
    if ( *(_QWORD *)(v11 + 8) >= v66 )
      return result;
  }
  *(_QWORD *)(v11 + 8) = v66;
  for ( result = *(_QWORD *)v8 + 176LL * *(unsigned int *)(v8 + 8);
        v11 + 176 != result;
        result = v39 + 176LL * *(unsigned int *)(v8 + 8) )
  {
    result = v66;
    if ( *(_QWORD *)(v11 + 176) > v66 )
      break;
    v33 = *(unsigned int *)(v11 + 216);
    v34 = *(unsigned int *)(v11 + 40);
    v35 = *(const void **)(v11 + 208);
    v36 = 8 * v33;
    if ( v33 + v34 > (unsigned __int64)*(unsigned int *)(v11 + 44) )
    {
      sub_C8D5F0(v62, (const void *)(v11 + 48), v33 + v34, 8u, v36, v10);
      v34 = *(unsigned int *)(v11 + 40);
      v36 = 8 * v33;
    }
    if ( v36 )
    {
      memcpy((void *)(*(_QWORD *)(v11 + 32) + 8 * v34), v35, v36);
      LODWORD(v34) = *(_DWORD *)(v11 + 40);
    }
    v37 = v34 + v33;
    v38 = *(_QWORD *)(v11 + 184);
    *(_DWORD *)(v11 + 40) = v37;
    if ( v38 > *(_QWORD *)(v11 + 8) )
      *(_QWORD *)(v11 + 8) = v38;
    v39 = *(_QWORD *)v8;
    v40 = *(_DWORD *)(v8 + 8);
    v41 = *(_QWORD *)v8 - v11 + 176LL * v40 - 352;
    v42 = 0x2E8BA2E8BA2E8BA3LL * (v41 >> 4);
    if ( v41 > 0 )
    {
      v43 = v11 + 208;
      do
      {
        v44 = *(_QWORD *)(v43 + 144);
        v45 = v43;
        v43 += 176;
        *(_QWORD *)(v43 - 208) = v44;
        *(_QWORD *)(v43 - 200) = *(_QWORD *)(v43 - 24);
        *(_QWORD *)(v43 - 192) = *(_QWORD *)(v43 - 16);
        v46 = *(unsigned __int16 *)(v43 - 8);
        *(_WORD *)(v43 - 184) = v46;
        sub_28A9600(v45, (char **)v43, v46, v41, v36, v10);
        --v42;
      }
      while ( v42 );
      v40 = *(_DWORD *)(v8 + 8);
      v39 = *(_QWORD *)v8;
    }
    v47 = v40 - 1;
    *(_DWORD *)(v8 + 8) = v47;
    v48 = v39 + 176 * v47;
    v49 = *(_QWORD *)(v48 + 32);
    if ( v49 != v48 + 48 )
    {
      _libc_free(v49);
      v39 = *(_QWORD *)v8;
    }
  }
  return result;
}
