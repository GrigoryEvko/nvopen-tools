// Function: sub_16BB3F0
// Address: 0x16bb3f0
//
__int64 __fastcall sub_16BB3F0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  size_t v10; // rsi
  size_t v11; // rdx
  _DWORD *v12; // rdi
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  int v16; // r13d
  __int64 v17; // rax
  _DWORD *v18; // r8
  __int64 v19; // rcx
  _DWORD *v20; // r14
  unsigned __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  _DWORD *v25; // r8
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _BOOL8 v28; // rdi
  __int64 v29; // rax
  int v30; // esi
  __int64 v31; // r15
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // ebx
  __int64 v36; // rax
  _DWORD *v37; // rdx
  _BOOL8 v38; // rdi
  _QWORD *v40; // rax
  unsigned __int64 v41; // [rsp+10h] [rbp-80h]
  _DWORD *v42; // [rsp+10h] [rbp-80h]
  _QWORD *v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  _DWORD *v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+18h] [rbp-78h]
  _QWORD *dest; // [rsp+20h] [rbp-70h]
  _QWORD v48[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v49; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v10 = (size_t)a7;
  dest = v48;
  LOBYTE(v48[0]) = 0;
  if ( a7 )
  {
    v49 = src;
    sub_16BA750((__int64 *)&v49, a7, (__int64)&a7[a8]);
    v11 = (size_t)v49;
    if ( v49 != src )
    {
      v10 = n;
      dest = v49;
      v48[0] = src[0];
      v49 = src;
      v40 = src;
      goto LABEL_4;
    }
    v11 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        LOBYTE(v48[0]) = src[0];
      }
      else
      {
        v10 = (size_t)src;
        memcpy(v48, src, n);
      }
      v11 = n;
    }
  }
  else
  {
    LOBYTE(src[0]) = 0;
    v11 = 0;
    v49 = src;
  }
  *((_BYTE *)v48 + v11) = 0;
  v40 = v49;
LABEL_4:
  n = 0;
  *(_BYTE *)v40 = 0;
  v12 = v49;
  if ( v49 != src )
  {
    v10 = src[0] + 1LL;
    j_j___libc_free_0(v49, src[0] + 1LL);
  }
  *(_QWORD *)(a1 + 176) -= 4LL;
  *(_DWORD *)(a1 + 16) = a2;
  v13 = sub_16D5D50(v12, v10, v11);
  v14 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v12 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v10 = v14[2];
        v15 = v14[3];
        if ( v13 <= v14[4] )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v15 )
          goto LABEL_11;
      }
      v12 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v10 );
LABEL_11:
    v16 = -1;
    if ( v12 != dword_4FA0208 && v13 >= *((_QWORD *)v12 + 4) )
    {
      v17 = *((_QWORD *)v12 + 7);
      v18 = v12 + 12;
      if ( v17 )
      {
        v10 = *(unsigned int *)(a1 + 8);
        v12 += 12;
        do
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v17 + 16);
            v13 = *(_QWORD *)(v17 + 24);
            if ( *(_DWORD *)(v17 + 32) >= (int)v10 )
              break;
            v17 = *(_QWORD *)(v17 + 24);
            if ( !v13 )
              goto LABEL_18;
          }
          v12 = (_DWORD *)v17;
          v17 = *(_QWORD *)(v17 + 16);
        }
        while ( v19 );
LABEL_18:
        v16 = -1;
        if ( v12 != v18 && (int)v10 >= v12[8] )
          v16 = v12[9] - 1;
      }
    }
  }
  else
  {
    v16 = -1;
  }
  v20 = dword_4FA0208;
  v21 = sub_16D5D50(v12, v10, v13);
  v22 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_28;
  do
  {
    while ( 1 )
    {
      v23 = v22[2];
      v24 = v22[3];
      if ( v21 <= v22[4] )
        break;
      v22 = (_QWORD *)v22[3];
      if ( !v24 )
        goto LABEL_26;
    }
    v20 = v22;
    v22 = (_QWORD *)v22[2];
  }
  while ( v23 );
LABEL_26:
  if ( v20 == dword_4FA0208 || (v25 = v20 + 12, v21 < *((_QWORD *)v20 + 4)) )
  {
LABEL_28:
    v41 = v21;
    v43 = v20;
    v20 = (_DWORD *)sub_22077B0(88);
    v20[12] = 0;
    *((_QWORD *)v20 + 4) = v41;
    *((_QWORD *)v20 + 7) = 0;
    *((_QWORD *)v20 + 8) = v20 + 12;
    *((_QWORD *)v20 + 9) = v20 + 12;
    *((_QWORD *)v20 + 10) = 0;
    v26 = sub_981350(&qword_4FA0200, v43, (unsigned __int64 *)v20 + 4);
    if ( v27 )
    {
      v28 = v26 || dword_4FA0208 == (_DWORD *)v27 || v41 < *(_QWORD *)(v27 + 32);
      sub_220F040(v28, v20, v27, dword_4FA0208);
      ++*(_QWORD *)&dword_4FA0208[8];
      v25 = v20 + 12;
    }
    else
    {
      v45 = v26;
      sub_16BA920(0);
      j_j___libc_free_0(v20, 88);
      v25 = v45 + 12;
      v20 = v45;
    }
  }
  v29 = *((_QWORD *)v20 + 7);
  if ( !v29 )
  {
    v31 = (__int64)v25;
LABEL_40:
    v44 = v31;
    v42 = v25;
    v34 = sub_22077B0(40);
    v35 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(v34 + 36) = 0;
    v31 = v34;
    *(_DWORD *)(v34 + 32) = v35;
    v36 = sub_9814F0((_QWORD *)v20 + 5, v44, (int *)(v34 + 32));
    if ( v37 )
    {
      v38 = v42 == v37 || v36 || v35 < v37[8];
      sub_220F040(v38, v31, v37, v42);
      ++*((_QWORD *)v20 + 10);
    }
    else
    {
      v46 = v36;
      j_j___libc_free_0(v31, 40);
      v31 = v46;
    }
    goto LABEL_45;
  }
  v30 = *(_DWORD *)(a1 + 8);
  v31 = (__int64)v25;
  do
  {
    while ( 1 )
    {
      v32 = *(_QWORD *)(v29 + 16);
      v33 = *(_QWORD *)(v29 + 24);
      if ( *(_DWORD *)(v29 + 32) >= v30 )
        break;
      v29 = *(_QWORD *)(v29 + 24);
      if ( !v33 )
        goto LABEL_38;
    }
    v31 = v29;
    v29 = *(_QWORD *)(v29 + 16);
  }
  while ( v32 );
LABEL_38:
  if ( (_DWORD *)v31 == v25 || v30 < *(_DWORD *)(v31 + 32) )
    goto LABEL_40;
LABEL_45:
  *(_DWORD *)(v31 + 36) = v16;
  if ( dest != v48 )
    j_j___libc_free_0(dest, v48[0] + 1LL);
  return 0;
}
