// Function: sub_BBA300
// Address: 0xbba300
//
__int64 __fastcall sub_BBA300(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  _QWORD *v10; // rsi
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // r8
  __int64 v23; // rcx
  int v24; // r13d
  _QWORD *v25; // r14
  unsigned __int64 v26; // rcx
  _QWORD *v27; // rax
  __int64 v28; // r15
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // r8
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  _BOOL8 v34; // rdi
  __int64 v35; // rax
  int v36; // esi
  __int64 v37; // r14
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // ebx
  __int64 v42; // rax
  __int64 v43; // rdx
  _BOOL8 v44; // rdi
  _QWORD *v46; // rax
  size_t v47; // rdx
  _QWORD *v48; // [rsp+8h] [rbp-88h]
  unsigned __int64 v49; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  _QWORD *v51; // [rsp+18h] [rbp-78h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  _QWORD *v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  _QWORD *dest; // [rsp+20h] [rbp-70h]
  _QWORD v56[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v57; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v10 = a7;
  dest = v56;
  LOBYTE(v56[0]) = 0;
  if ( a7 )
  {
    v57 = src;
    sub_BB88C0((__int64 *)&v57, a7, (__int64)&a7[a8]);
    if ( v57 != src )
    {
      v10 = (_QWORD *)n;
      dest = v57;
      v56[0] = src[0];
      v57 = src;
      v46 = src;
      goto LABEL_4;
    }
    v47 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        LOBYTE(v56[0]) = src[0];
      }
      else
      {
        v10 = src;
        memcpy(v56, src, n);
      }
      v47 = n;
    }
  }
  else
  {
    LOBYTE(src[0]) = 0;
    v47 = 0;
    v57 = src;
  }
  *((_BYTE *)v56 + v47) = 0;
  v46 = v57;
LABEL_4:
  n = 0;
  *(_BYTE *)v46 = 0;
  if ( v57 != src )
  {
    v10 = (_QWORD *)(src[0] + 1LL);
    j_j___libc_free_0(v57, src[0] + 1LL);
  }
  v11 = *(_QWORD *)(a1 + 144);
  *(_QWORD *)(a1 + 144) = v11 - 32;
  v12 = *(_QWORD **)(v11 - 32);
  v13 = v11 - 16;
  if ( v12 != (_QWORD *)(v11 - 16) )
  {
    v10 = (_QWORD *)(*(_QWORD *)(v11 - 16) + 1LL);
    j_j___libc_free_0(v12, v10);
  }
  *(_QWORD *)(a1 + 200) -= 4LL;
  *(_WORD *)(a1 + 14) = a2;
  v14 = sub_C52410(v12, v10, v13);
  v15 = v14 + 8;
  v17 = sub_C959E0();
  v18 = *(_QWORD **)(v14 + 16);
  if ( v18 )
  {
    v12 = (_QWORD *)(v14 + 8);
    do
    {
      while ( 1 )
      {
        v19 = v18[2];
        v16 = v18[3];
        if ( v17 <= v18[4] )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v16 )
          goto LABEL_13;
      }
      v12 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v19 );
LABEL_13:
    if ( (_QWORD *)v15 != v12 && v17 >= v12[4] )
      v15 = (__int64)v12;
  }
  if ( v15 == sub_C52410(v12, v17, v16) + 8 || (v21 = *(_QWORD *)(v15 + 56), v22 = (_QWORD *)(v15 + 48), !v21) )
  {
    v24 = -1;
  }
  else
  {
    v17 = *(unsigned int *)(a1 + 8);
    v12 = (_QWORD *)(v15 + 48);
    do
    {
      while ( 1 )
      {
        v23 = *(_QWORD *)(v21 + 16);
        v20 = *(_QWORD *)(v21 + 24);
        if ( *(_DWORD *)(v21 + 32) >= (int)v17 )
          break;
        v21 = *(_QWORD *)(v21 + 24);
        if ( !v20 )
          goto LABEL_22;
      }
      v12 = (_QWORD *)v21;
      v21 = *(_QWORD *)(v21 + 16);
    }
    while ( v23 );
LABEL_22:
    v24 = -1;
    if ( v12 != v22 && (int)v17 >= *((_DWORD *)v12 + 8) )
      v24 = *((_DWORD *)v12 + 9) - 1;
  }
  v25 = (_QWORD *)sub_C52410(v12, v17, v20);
  v26 = sub_C959E0();
  v27 = (_QWORD *)v25[2];
  v28 = (__int64)(v25 + 1);
  if ( !v27 )
    goto LABEL_32;
  do
  {
    while ( 1 )
    {
      v29 = v27[2];
      v30 = v27[3];
      if ( v26 <= v27[4] )
        break;
      v27 = (_QWORD *)v27[3];
      if ( !v30 )
        goto LABEL_30;
    }
    v28 = (__int64)v27;
    v27 = (_QWORD *)v27[2];
  }
  while ( v29 );
LABEL_30:
  if ( (_QWORD *)v28 == v25 + 1 || (v31 = v28 + 48, v26 < *(_QWORD *)(v28 + 32)) )
  {
LABEL_32:
    v49 = v26;
    v51 = (_QWORD *)v28;
    v48 = v25 + 1;
    v28 = sub_22077B0(88);
    *(_DWORD *)(v28 + 48) = 0;
    *(_QWORD *)(v28 + 32) = v49;
    *(_QWORD *)(v28 + 56) = 0;
    *(_QWORD *)(v28 + 64) = v28 + 48;
    *(_QWORD *)(v28 + 72) = v28 + 48;
    *(_QWORD *)(v28 + 80) = 0;
    v32 = sub_981350(v25, v51, (unsigned __int64 *)(v28 + 32));
    if ( v33 )
    {
      v34 = v32 || v48 == v33 || v49 < v33[4];
      sub_220F040(v34, v28, v33, v48);
      ++v25[5];
      v31 = v28 + 48;
    }
    else
    {
      v53 = v32;
      sub_BB8C90(0);
      j_j___libc_free_0(v28, 88);
      v31 = (__int64)(v53 + 6);
      v28 = (__int64)v53;
    }
  }
  v35 = *(_QWORD *)(v28 + 56);
  if ( !v35 )
  {
    v37 = v31;
LABEL_44:
    v52 = v37;
    v50 = v31;
    v40 = sub_22077B0(40);
    v41 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(v40 + 36) = 0;
    v37 = v40;
    *(_DWORD *)(v40 + 32) = v41;
    v42 = sub_9814F0((_QWORD *)(v28 + 40), v52, (int *)(v40 + 32));
    if ( v43 )
    {
      v44 = v50 == v43 || v42 || v41 < *(_DWORD *)(v43 + 32);
      sub_220F040(v44, v37, v43, v50);
      ++*(_QWORD *)(v28 + 80);
    }
    else
    {
      v54 = v42;
      j_j___libc_free_0(v37, 40);
      v37 = v54;
    }
    goto LABEL_49;
  }
  v36 = *(_DWORD *)(a1 + 8);
  v37 = v31;
  do
  {
    while ( 1 )
    {
      v38 = *(_QWORD *)(v35 + 16);
      v39 = *(_QWORD *)(v35 + 24);
      if ( *(_DWORD *)(v35 + 32) >= v36 )
        break;
      v35 = *(_QWORD *)(v35 + 24);
      if ( !v39 )
        goto LABEL_42;
    }
    v37 = v35;
    v35 = *(_QWORD *)(v35 + 16);
  }
  while ( v38 );
LABEL_42:
  if ( v37 == v31 || v36 < *(_DWORD *)(v37 + 32) )
    goto LABEL_44;
LABEL_49:
  *(_DWORD *)(v37 + 36) = v24;
  if ( dest != v56 )
    j_j___libc_free_0(dest, v56[0] + 1LL);
  return 0;
}
