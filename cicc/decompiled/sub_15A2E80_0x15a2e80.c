// Function: sub_15A2E80
// Address: 0x15a2e80
//
__int64 __fastcall sub_15A2E80(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        unsigned __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v8; // r13
  __int64 v11; // r9
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 **v17; // rcx
  __int64 **v18; // rax
  unsigned int v19; // r8d
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned int v23; // r8d
  __int64 *v24; // rcx
  __int64 v25; // rbx
  _BYTE *v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rax
  char v29; // r12
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // r9
  _BYTE *v38; // rcx
  _BYTE *v39; // rax
  unsigned int v40; // r8d
  _BYTE *v41; // [rsp+0h] [rbp-B0h]
  __int64 *v42; // [rsp+8h] [rbp-A8h]
  __int64 *v43; // [rsp+8h] [rbp-A8h]
  unsigned int v44; // [rsp+8h] [rbp-A8h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  unsigned int v46; // [rsp+10h] [rbp-A0h]
  unsigned int v47; // [rsp+10h] [rbp-A0h]
  unsigned int v48; // [rsp+10h] [rbp-A0h]
  unsigned int v49; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+10h] [rbp-A0h]
  unsigned int v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __int64 v54; // [rsp+20h] [rbp-90h]
  __int64 v55; // [rsp+28h] [rbp-88h] BYREF
  void *src; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v57; // [rsp+38h] [rbp-78h]
  _BYTE *v58; // [rsp+40h] [rbp-70h]
  __int128 v59; // [rsp+50h] [rbp-60h] BYREF
  __int128 v60; // [rsp+60h] [rbp-50h]
  __int128 v61; // [rsp+70h] [rbp-40h]

  v8 = a1;
  v55 = a2;
  if ( !a1 )
  {
    v13 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v13 = **(_QWORD **)(v13 + 16);
    v8 = *(_QWORD *)(v13 + 24);
  }
  BYTE4(v59) = *(_BYTE *)(a6 + 4);
  if ( BYTE4(v59) )
    LODWORD(v59) = *(_DWORD *)a6;
  v11 = sub_1588490(v8, a2, a5, (int *)&v59, a3, a4);
  if ( v11 )
    return v11;
  v14 = sub_15F9F50(v8, a3, a4);
  v15 = *(_QWORD *)v55;
  if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) == 16 )
    v15 = **(_QWORD **)(v15 + 16);
  v16 = sub_1647190(v14, *(_DWORD *)(v15 + 8) >> 8);
  v11 = 0;
  v52 = v16;
  if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) == 16 )
  {
    v19 = *(_DWORD *)(*(_QWORD *)v55 + 32LL);
  }
  else
  {
    v17 = &a3[a4];
    if ( a3 == v17 )
    {
      v19 = 0;
      goto LABEL_19;
    }
    v18 = a3;
    v19 = 0;
    do
    {
      v20 = **v18;
      if ( *(_BYTE *)(v20 + 8) == 16 )
        v19 = *(_DWORD *)(v20 + 32);
      ++v18;
    }
    while ( v17 != v18 );
  }
  if ( v19 )
  {
    v46 = v19;
    v21 = sub_16463B0(v52, v19);
    v19 = v46;
    v11 = 0;
    v52 = v21;
  }
LABEL_19:
  if ( v52 != a7 )
  {
    v22 = a4 + 1;
    src = 0;
    v57 = 0;
    v58 = 0;
    if ( a4 + 1 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::reserve");
    if ( a4 == -1 )
    {
      v47 = v19;
      sub_127D720((__int64)&src, 0, &v55);
      v23 = v47;
    }
    else
    {
      v44 = v19;
      v50 = 8 * v22;
      v36 = (_BYTE *)sub_22077B0(8 * v22);
      v37 = v50;
      v23 = v44;
      v38 = v36;
      if ( v57 - (_BYTE *)src > 0 )
      {
        v39 = memmove(v36, src, v57 - (_BYTE *)src);
        v40 = v44;
        v41 = v39;
        v45 = v50;
        v51 = v40;
        j_j___libc_free_0(src, v58 - (_BYTE *)src);
        v38 = v41;
        v37 = v45;
        v23 = v51;
      }
      src = v38;
      v58 = &v38[v37];
      if ( v38 )
        *(_QWORD *)v38 = v55;
      v57 = v38 + 8;
    }
    if ( (_DWORD)a4 )
    {
      v24 = (__int64 *)a3;
      v25 = (__int64)&a3[(unsigned int)(a4 - 1) + 1];
      do
      {
        while ( 1 )
        {
          v27 = *v24;
          *(_QWORD *)&v59 = *v24;
          if ( v23 && *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 )
          {
            v42 = v24;
            v48 = v23;
            v28 = sub_15A0390(v23, v27);
            v24 = v42;
            v23 = v48;
            *(_QWORD *)&v59 = v28;
          }
          v26 = v57;
          if ( v57 != v58 )
            break;
          v43 = v24;
          v49 = v23;
          sub_127D720((__int64)&src, v57, &v59);
          v23 = v49;
          v24 = v43 + 1;
          if ( (__int64 *)v25 == v43 + 1 )
            goto LABEL_33;
        }
        if ( v57 )
        {
          *(_QWORD *)v57 = v59;
          v26 = v57;
        }
        ++v24;
        v57 = v26 + 8;
      }
      while ( (__int64 *)v25 != v24 );
    }
LABEL_33:
    v29 = a5;
    if ( *(_BYTE *)(a6 + 4) && *(_DWORD *)a6 <= 0x3Eu )
      v29 = (2 * *(_DWORD *)a6 + 2) | a5;
    LOBYTE(v59) = 32;
    WORD1(v59) = 0;
    *((_QWORD *)&v59 + 1) = src;
    BYTE1(v59) = v29;
    v60 = (unsigned __int64)((v57 - (_BYTE *)src) >> 3);
    *(_QWORD *)&v61 = 0;
    *((_QWORD *)&v61 + 1) = v8;
    v30 = (_QWORD *)sub_16498A0(v55);
    v35 = sub_15A2780(*v30 + 1776LL, v52, v31, v32, v33, v34, v59, v60, v61);
    v11 = v35;
    if ( src )
    {
      v54 = v35;
      j_j___libc_free_0(src, v58 - (_BYTE *)src);
      return v54;
    }
  }
  return v11;
}
