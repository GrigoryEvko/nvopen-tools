// Function: sub_E728D0
// Address: 0xe728d0
//
__int64 __fastcall sub_E728D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  char v8; // al
  _QWORD *v9; // r12
  _QWORD *v10; // r15
  _QWORD *v11; // rdi
  __int64 v12; // rdi
  int v13; // eax
  _QWORD *v14; // r15
  _QWORD *v15; // rdx
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  int v20; // eax
  _QWORD *v22; // r15
  _QWORD *v23; // rdi
  __int64 v24; // rdi
  int v25; // eax
  _QWORD *v26; // r15
  _QWORD *v27; // rcx
  _QWORD *v28; // rdx
  _QWORD *v29; // r12
  _QWORD *v30; // rdi
  __int64 v31; // rdi
  int v32; // eax
  __int64 v33; // r15
  __int64 v34; // [rsp+0h] [rbp-60h]
  unsigned __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  _QWORD *v42; // [rsp+20h] [rbp-40h]
  _QWORD *v43; // [rsp+28h] [rbp-38h]
  _QWORD *v44; // [rsp+28h] [rbp-38h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]

  if ( a2 != a1 )
  {
    v7 = a1;
    while ( a4 != a3 )
    {
      v8 = sub_E72550(a3, v7);
      v9 = *(_QWORD **)(a5 + 40);
      v43 = *(_QWORD **)(a5 + 32);
      v40 = *(_QWORD *)(a5 + 48);
      if ( v8 )
      {
        *(_QWORD *)a5 = *(_QWORD *)a3;
        *(_QWORD *)(a5 + 8) = *(_QWORD *)(a3 + 8);
        *(_QWORD *)(a5 + 16) = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a5 + 24) = *(_QWORD *)(a3 + 24);
        *(_QWORD *)(a5 + 32) = *(_QWORD *)(a3 + 32);
        *(_QWORD *)(a5 + 40) = *(_QWORD *)(a3 + 40);
        *(_QWORD *)(a5 + 48) = *(_QWORD *)(a3 + 48);
        *(_QWORD *)(a3 + 32) = 0;
        *(_QWORD *)(a3 + 40) = 0;
        v10 = v43;
        for ( *(_QWORD *)(a3 + 48) = 0; v9 != v10; v10 += 13 )
        {
          v11 = (_QWORD *)v10[9];
          if ( v11 != v10 + 11 )
            j_j___libc_free_0(v11, v10[11] + 1LL);
          v12 = v10[6];
          if ( v12 )
            j_j___libc_free_0(v12, v10[8] - v12);
        }
        if ( v43 )
          j_j___libc_free_0(v43, v40 - (_QWORD)v43);
        v13 = *(_DWORD *)(a3 + 56);
        a3 += 96;
        *(_DWORD *)(a5 + 56) = v13;
        *(_DWORD *)(a5 + 60) = *(_DWORD *)(a3 - 36);
        *(_DWORD *)(a5 + 64) = *(_DWORD *)(a3 - 32);
        *(_QWORD *)(a5 + 72) = *(_QWORD *)(a3 - 24);
        *(_BYTE *)(a5 + 80) = *(_BYTE *)(a3 - 16);
        *(_BYTE *)(a5 + 81) = *(_BYTE *)(a3 - 15);
        *(_DWORD *)(a5 + 84) = *(_DWORD *)(a3 - 12);
        *(_BYTE *)(a5 + 88) = *(_BYTE *)(a3 - 8);
        *(_BYTE *)(a5 + 89) = *(_BYTE *)(a3 - 7);
      }
      else
      {
        *(_QWORD *)a5 = *(_QWORD *)v7;
        *(_QWORD *)(a5 + 8) = *(_QWORD *)(v7 + 8);
        *(_QWORD *)(a5 + 16) = *(_QWORD *)(v7 + 16);
        *(_QWORD *)(a5 + 24) = *(_QWORD *)(v7 + 24);
        *(_QWORD *)(a5 + 32) = *(_QWORD *)(v7 + 32);
        *(_QWORD *)(a5 + 40) = *(_QWORD *)(v7 + 40);
        *(_QWORD *)(a5 + 48) = *(_QWORD *)(v7 + 48);
        *(_QWORD *)(v7 + 32) = 0;
        *(_QWORD *)(v7 + 40) = 0;
        v22 = v43;
        for ( *(_QWORD *)(v7 + 48) = 0; v9 != v22; v22 += 13 )
        {
          v23 = (_QWORD *)v22[9];
          if ( v23 != v22 + 11 )
            j_j___libc_free_0(v23, v22[11] + 1LL);
          v24 = v22[6];
          if ( v24 )
            j_j___libc_free_0(v24, v22[8] - v24);
        }
        if ( v43 )
          j_j___libc_free_0(v43, v40 - (_QWORD)v43);
        v25 = *(_DWORD *)(v7 + 56);
        v7 += 96;
        *(_DWORD *)(a5 + 56) = v25;
        *(_DWORD *)(a5 + 60) = *(_DWORD *)(v7 - 36);
        *(_DWORD *)(a5 + 64) = *(_DWORD *)(v7 - 32);
        *(_QWORD *)(a5 + 72) = *(_QWORD *)(v7 - 24);
        *(_BYTE *)(a5 + 80) = *(_BYTE *)(v7 - 16);
        *(_BYTE *)(a5 + 81) = *(_BYTE *)(v7 - 15);
        *(_DWORD *)(a5 + 84) = *(_DWORD *)(v7 - 12);
        *(_BYTE *)(a5 + 88) = *(_BYTE *)(v7 - 8);
        *(_BYTE *)(a5 + 89) = *(_BYTE *)(v7 - 7);
      }
      a5 += 96;
      if ( a2 == v7 )
        goto LABEL_15;
    }
    v34 = a2 - v7;
    v35 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v7) >> 5);
    if ( a2 - v7 <= 0 )
      return a5;
    v26 = (_QWORD *)a5;
    do
    {
      v27 = (_QWORD *)v26[4];
      v28 = (_QWORD *)v26[5];
      *v26 = *(_QWORD *)v7;
      v29 = v27;
      v42 = v27;
      v26[1] = *(_QWORD *)(v7 + 8);
      v45 = v28;
      v26[2] = *(_QWORD *)(v7 + 16);
      v26[3] = *(_QWORD *)(v7 + 24);
      v39 = v26[6];
      v26[4] = *(_QWORD *)(v7 + 32);
      v26[5] = *(_QWORD *)(v7 + 40);
      v26[6] = *(_QWORD *)(v7 + 48);
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = 0;
      for ( *(_QWORD *)(v7 + 48) = 0; v45 != v29; v29 += 13 )
      {
        v30 = (_QWORD *)v29[9];
        if ( v30 != v29 + 11 )
          j_j___libc_free_0(v30, v29[11] + 1LL);
        v31 = v29[6];
        if ( v31 )
          j_j___libc_free_0(v31, v29[8] - v31);
      }
      if ( v42 )
        j_j___libc_free_0(v42, v39 - (_QWORD)v42);
      v32 = *(_DWORD *)(v7 + 56);
      v26 += 12;
      v7 += 96;
      *((_DWORD *)v26 - 10) = v32;
      *((_DWORD *)v26 - 9) = *(_DWORD *)(v7 - 36);
      *((_DWORD *)v26 - 8) = *(_DWORD *)(v7 - 32);
      *(v26 - 3) = *(_QWORD *)(v7 - 24);
      *((_BYTE *)v26 - 16) = *(_BYTE *)(v7 - 16);
      *((_BYTE *)v26 - 15) = *(_BYTE *)(v7 - 15);
      *((_DWORD *)v26 - 3) = *(_DWORD *)(v7 - 12);
      *((_BYTE *)v26 - 8) = *(_BYTE *)(v7 - 8);
      *((_BYTE *)v26 - 7) = *(_BYTE *)(v7 - 7);
      --v35;
    }
    while ( v35 );
    v33 = 96;
    if ( v34 > 0 )
      v33 = v34;
    a5 += v33;
  }
LABEL_15:
  v37 = a4 - a3;
  v38 = 0xAAAAAAAAAAAAAAABLL * (v37 >> 5);
  if ( v37 <= 0 )
    return a5;
  v14 = (_QWORD *)a5;
  do
  {
    v15 = (_QWORD *)v14[4];
    v16 = (_QWORD *)v14[5];
    *v14 = *(_QWORD *)a3;
    v17 = v15;
    v44 = v15;
    v14[1] = *(_QWORD *)(a3 + 8);
    v14[2] = *(_QWORD *)(a3 + 16);
    v14[3] = *(_QWORD *)(a3 + 24);
    v41 = v14[6];
    v14[4] = *(_QWORD *)(a3 + 32);
    v14[5] = *(_QWORD *)(a3 + 40);
    v14[6] = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(a3 + 32) = 0;
    *(_QWORD *)(a3 + 40) = 0;
    for ( *(_QWORD *)(a3 + 48) = 0; v16 != v17; v17 += 13 )
    {
      v18 = (_QWORD *)v17[9];
      if ( v18 != v17 + 11 )
        j_j___libc_free_0(v18, v17[11] + 1LL);
      v19 = v17[6];
      if ( v19 )
        j_j___libc_free_0(v19, v17[8] - v19);
    }
    if ( v44 )
      j_j___libc_free_0(v44, v41 - (_QWORD)v44);
    v20 = *(_DWORD *)(a3 + 56);
    v14 += 12;
    a3 += 96;
    *((_DWORD *)v14 - 10) = v20;
    *((_DWORD *)v14 - 9) = *(_DWORD *)(a3 - 36);
    *((_DWORD *)v14 - 8) = *(_DWORD *)(a3 - 32);
    *(v14 - 3) = *(_QWORD *)(a3 - 24);
    *((_BYTE *)v14 - 16) = *(_BYTE *)(a3 - 16);
    *((_BYTE *)v14 - 15) = *(_BYTE *)(a3 - 15);
    *((_DWORD *)v14 - 3) = *(_DWORD *)(a3 - 12);
    *((_BYTE *)v14 - 8) = *(_BYTE *)(a3 - 8);
    *((_BYTE *)v14 - 7) = *(_BYTE *)(a3 - 7);
    --v38;
  }
  while ( v38 );
  return a5 + v37;
}
