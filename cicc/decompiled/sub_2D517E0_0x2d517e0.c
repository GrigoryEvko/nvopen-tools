// Function: sub_2D517E0
// Address: 0x2d517e0
//
__int64 __fastcall sub_2D517E0(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  int v6; // eax
  int v7; // eax
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  int v20; // esi
  unsigned int v21; // r15d
  __int64 v23; // rcx
  unsigned __int64 v24; // r8
  __int64 v25; // rsi
  unsigned __int64 *v26; // rbx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 *v30; // r13
  char **v31; // rbx
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // rbx
  unsigned __int64 *v35; // rbx
  __int64 v36; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v39[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v40[48]; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 *v41; // [rsp+60h] [rbp-80h] BYREF
  __int64 v42; // [rsp+68h] [rbp-78h]
  _BYTE v43[64]; // [rsp+70h] [rbp-70h] BYREF
  char v44; // [rsp+B0h] [rbp-30h] BYREF

  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a2 + 120), a3, a4, v6);
  if ( v7 != -1 )
  {
    v8 = *(_QWORD *)(a2 + 120);
    v9 = (__int64 *)(v8 + 8LL * v7);
    if ( v9 != (__int64 *)(v8 + 8LL * *(unsigned int *)(a2 + 128)) )
    {
      v10 = *v9;
      a3 = *(const void **)(v10 + 8);
      a4 = *(_QWORD *)(v10 + 16);
    }
  }
  v11 = sub_C92610();
  v12 = sub_C92860((__int64 *)(a2 + 96), a3, a4, v11);
  if ( v12 == -1
    || (v15 = *(_QWORD *)(a2 + 96),
        v16 = *(unsigned int *)(a2 + 104),
        v17 = (__int64 *)(v15 + 8LL * v12),
        v18 = v15 + 8 * v16,
        v17 == (__int64 *)v18) )
  {
    *(_QWORD *)a1 = a1 + 16;
    v39[0] = (unsigned __int64)v40;
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    goto LABEL_10;
  }
  v19 = *v17;
  v39[0] = (unsigned __int64)v40;
  v20 = *(_DWORD *)(v19 + 16);
  v39[1] = 0x300000000LL;
  if ( v20 )
    sub_2D50040((__int64)v39, v19 + 8, v18, v16, v13, v14);
  v21 = *(_DWORD *)(v19 + 80);
  v41 = (unsigned __int64 *)v43;
  v42 = 0x100000000LL;
  if ( !v21 || (v23 = v19 + 72, &v41 == (unsigned __int64 **)(v19 + 72)) )
  {
    *(_QWORD *)(a1 + 8) = 0x100000000LL;
    *(_QWORD *)a1 = a1 + 16;
    goto LABEL_10;
  }
  v24 = v21;
  if ( v21 == 1 )
  {
    v25 = *(_QWORD *)(v19 + 72);
    v26 = (unsigned __int64 *)v43;
    v27 = v25 + 64;
    v28 = v25 + 64;
  }
  else
  {
    sub_2D516E0((__int64)&v41, v21, v18, v23, v21, v14);
    v25 = *(_QWORD *)(v19 + 72);
    v26 = v41;
    v24 = v21;
    v28 = v25 + ((unsigned __int64)*(unsigned int *)(v19 + 80) << 6);
    if ( v25 == v28 )
      goto LABEL_25;
    v27 = v25 + 64;
  }
  while ( 1 )
  {
    if ( v26 )
    {
      *((_DWORD *)v26 + 2) = 0;
      *v26 = (unsigned __int64)(v26 + 2);
      *((_DWORD *)v26 + 3) = 12;
      v23 = *(unsigned int *)(v25 + 8);
      if ( (_DWORD)v23 )
      {
        v36 = v28;
        v37 = v24;
        sub_2D50120((__int64)v26, v25, v18, v23, v24, v14);
        v28 = v36;
        v24 = v37;
      }
    }
    v25 = v27;
    v26 += 8;
    if ( v27 == v28 )
      break;
    v27 += 64;
  }
  v26 = v41;
LABEL_25:
  v29 = a1 + 16;
  LODWORD(v42) = v21;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  if ( v26 != (unsigned __int64 *)v43 )
  {
    v33 = HIDWORD(v42);
    *(_QWORD *)a1 = v26;
    *(_DWORD *)(a1 + 8) = v21;
    *(_DWORD *)(a1 + 12) = v33;
    goto LABEL_10;
  }
  v30 = (unsigned __int64 *)&v44;
  v31 = (char **)v43;
  if ( v21 == 1 )
    goto LABEL_29;
  sub_2D516E0(a1, v24, v18, v23, v24, v14);
  v31 = (char **)v41;
  v30 = &v41[8 * (unsigned __int64)(unsigned int)v42];
  if ( v41 != v30 )
  {
    v29 = *(_QWORD *)a1;
    do
    {
LABEL_29:
      if ( v29 )
      {
        *(_DWORD *)(v29 + 8) = 0;
        *(_QWORD *)v29 = v29 + 16;
        *(_DWORD *)(v29 + 12) = 12;
        v32 = *((unsigned int *)v31 + 2);
        if ( (_DWORD)v32 )
        {
          v38 = v29;
          sub_2D50200(v29, v31, v32, v29 + 16, v24, v14);
          v29 = v38;
        }
      }
      v31 += 8;
      v29 += 64;
    }
    while ( v30 != (unsigned __int64 *)v31 );
    v34 = (unsigned int)v42;
    v30 = v41;
    *(_DWORD *)(a1 + 8) = v21;
    v35 = &v30[8 * v34];
    if ( v35 != v30 )
    {
      do
      {
        v35 -= 8;
        if ( (unsigned __int64 *)*v35 != v35 + 2 )
          _libc_free(*v35);
      }
      while ( v35 != v30 );
      v30 = v41;
    }
    goto LABEL_35;
  }
  *(_DWORD *)(a1 + 8) = v21;
LABEL_35:
  if ( v30 != (unsigned __int64 *)v43 )
    _libc_free((unsigned __int64)v30);
LABEL_10:
  if ( (_BYTE *)v39[0] != v40 )
    _libc_free(v39[0]);
  return a1;
}
