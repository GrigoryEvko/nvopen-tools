// Function: sub_E72E30
// Address: 0xe72e30
//
__int64 __fastcall sub_E72E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 i; // r13
  char v8; // al
  _QWORD *v9; // rcx
  _QWORD *v10; // rbx
  _QWORD *v11; // r15
  _QWORD *v12; // rdi
  __int64 v13; // rdi
  int v14; // eax
  _QWORD *v15; // r15
  _QWORD *v16; // rdx
  _QWORD *v17; // rcx
  _QWORD *v18; // rbx
  _QWORD *v19; // rdi
  __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // r15
  _QWORD *v23; // r15
  _QWORD *v24; // rdx
  _QWORD *v25; // rbx
  _QWORD *v26; // r13
  _QWORD *v27; // rdi
  __int64 v28; // rdi
  int v29; // eax
  _QWORD *v31; // r15
  _QWORD *v32; // rdi
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // [rsp+0h] [rbp-60h]
  unsigned __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h]
  _QWORD *v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  _QWORD *v44; // [rsp+28h] [rbp-38h]
  _QWORD *v45; // [rsp+28h] [rbp-38h]
  _QWORD *v46; // [rsp+28h] [rbp-38h]

  v5 = a3;
  for ( i = a1; i != a2 && v5 != a4; a5 += 96 )
  {
    v8 = sub_E72550(v5, i);
    v9 = *(_QWORD **)(a5 + 32);
    v10 = *(_QWORD **)(a5 + 40);
    v44 = v9;
    v41 = *(_QWORD *)(a5 + 48);
    if ( v8 )
    {
      v11 = *(_QWORD **)(a5 + 32);
      *(_QWORD *)a5 = *(_QWORD *)v5;
      *(_QWORD *)(a5 + 8) = *(_QWORD *)(v5 + 8);
      *(_QWORD *)(a5 + 16) = *(_QWORD *)(v5 + 16);
      *(_QWORD *)(a5 + 24) = *(_QWORD *)(v5 + 24);
      *(_QWORD *)(a5 + 32) = *(_QWORD *)(v5 + 32);
      *(_QWORD *)(a5 + 40) = *(_QWORD *)(v5 + 40);
      *(_QWORD *)(a5 + 48) = *(_QWORD *)(v5 + 48);
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_QWORD *)(v5 + 48) = 0;
      if ( v9 != v10 )
      {
        do
        {
          v12 = (_QWORD *)v11[9];
          if ( v12 != v11 + 11 )
            j_j___libc_free_0(v12, v11[11] + 1LL);
          v13 = v11[6];
          if ( v13 )
            j_j___libc_free_0(v13, v11[8] - v13);
          v11 += 13;
        }
        while ( v11 != v10 );
      }
      if ( v44 )
        j_j___libc_free_0(v44, v41 - (_QWORD)v44);
      v14 = *(_DWORD *)(v5 + 56);
      v5 += 96;
      *(_DWORD *)(a5 + 56) = v14;
      *(_DWORD *)(a5 + 60) = *(_DWORD *)(v5 - 36);
      *(_DWORD *)(a5 + 64) = *(_DWORD *)(v5 - 32);
      *(_QWORD *)(a5 + 72) = *(_QWORD *)(v5 - 24);
      *(_BYTE *)(a5 + 80) = *(_BYTE *)(v5 - 16);
      *(_BYTE *)(a5 + 81) = *(_BYTE *)(v5 - 15);
      *(_DWORD *)(a5 + 84) = *(_DWORD *)(v5 - 12);
      *(_BYTE *)(a5 + 88) = *(_BYTE *)(v5 - 8);
      *(_BYTE *)(a5 + 89) = *(_BYTE *)(v5 - 7);
    }
    else
    {
      *(_QWORD *)a5 = *(_QWORD *)i;
      *(_QWORD *)(a5 + 8) = *(_QWORD *)(i + 8);
      *(_QWORD *)(a5 + 16) = *(_QWORD *)(i + 16);
      *(_QWORD *)(a5 + 24) = *(_QWORD *)(i + 24);
      *(_QWORD *)(a5 + 32) = *(_QWORD *)(i + 32);
      *(_QWORD *)(a5 + 40) = *(_QWORD *)(i + 40);
      *(_QWORD *)(a5 + 48) = *(_QWORD *)(i + 48);
      *(_QWORD *)(i + 32) = 0;
      *(_QWORD *)(i + 40) = 0;
      v31 = v9;
      for ( *(_QWORD *)(i + 48) = 0; v31 != v10; v31 += 13 )
      {
        v32 = (_QWORD *)v31[9];
        if ( v32 != v31 + 11 )
          j_j___libc_free_0(v32, v31[11] + 1LL);
        v33 = v31[6];
        if ( v33 )
          j_j___libc_free_0(v33, v31[8] - v33);
      }
      if ( v44 )
        j_j___libc_free_0(v44, v41 - (_QWORD)v44);
      v34 = *(_DWORD *)(i + 56);
      i += 96;
      *(_DWORD *)(a5 + 56) = v34;
      *(_DWORD *)(a5 + 60) = *(_DWORD *)(i - 36);
      *(_DWORD *)(a5 + 64) = *(_DWORD *)(i - 32);
      *(_QWORD *)(a5 + 72) = *(_QWORD *)(i - 24);
      *(_BYTE *)(a5 + 80) = *(_BYTE *)(i - 16);
      *(_BYTE *)(a5 + 81) = *(_BYTE *)(i - 15);
      *(_DWORD *)(a5 + 84) = *(_DWORD *)(i - 12);
      *(_BYTE *)(a5 + 88) = *(_BYTE *)(i - 8);
      *(_BYTE *)(a5 + 89) = *(_BYTE *)(i - 7);
    }
  }
  v35 = a2 - i;
  v36 = 0xAAAAAAAAAAAAAAABLL * ((a2 - i) >> 5);
  if ( a2 - i > 0 )
  {
    v15 = (_QWORD *)a5;
    do
    {
      v16 = (_QWORD *)v15[4];
      v17 = (_QWORD *)v15[5];
      *v15 = *(_QWORD *)i;
      v18 = v16;
      v42 = v16;
      v15[1] = *(_QWORD *)(i + 8);
      v45 = v17;
      v15[2] = *(_QWORD *)(i + 16);
      v15[3] = *(_QWORD *)(i + 24);
      v39 = v15[6];
      v15[4] = *(_QWORD *)(i + 32);
      v15[5] = *(_QWORD *)(i + 40);
      v15[6] = *(_QWORD *)(i + 48);
      *(_QWORD *)(i + 32) = 0;
      *(_QWORD *)(i + 40) = 0;
      for ( *(_QWORD *)(i + 48) = 0; v45 != v18; v18 += 13 )
      {
        v19 = (_QWORD *)v18[9];
        if ( v19 != v18 + 11 )
          j_j___libc_free_0(v19, v18[11] + 1LL);
        v20 = v18[6];
        if ( v20 )
          j_j___libc_free_0(v20, v18[8] - v20);
      }
      if ( v42 )
        j_j___libc_free_0(v42, v39 - (_QWORD)v42);
      v21 = *(_DWORD *)(i + 56);
      v15 += 12;
      i += 96;
      *((_DWORD *)v15 - 10) = v21;
      *((_DWORD *)v15 - 9) = *(_DWORD *)(i - 36);
      *((_DWORD *)v15 - 8) = *(_DWORD *)(i - 32);
      *(v15 - 3) = *(_QWORD *)(i - 24);
      *((_BYTE *)v15 - 16) = *(_BYTE *)(i - 16);
      *((_BYTE *)v15 - 15) = *(_BYTE *)(i - 15);
      *((_DWORD *)v15 - 3) = *(_DWORD *)(i - 12);
      *((_BYTE *)v15 - 8) = *(_BYTE *)(i - 8);
      *((_BYTE *)v15 - 7) = *(_BYTE *)(i - 7);
      --v36;
    }
    while ( v36 );
    v22 = 96;
    if ( v35 > 0 )
      v22 = v35;
    a5 += v22;
  }
  v38 = a4 - v5;
  v40 = 0xAAAAAAAAAAAAAAABLL * (v38 >> 5);
  if ( v38 > 0 )
  {
    v23 = (_QWORD *)a5;
    do
    {
      v24 = (_QWORD *)v23[4];
      v25 = (_QWORD *)v23[5];
      *v23 = *(_QWORD *)v5;
      v26 = v24;
      v46 = v24;
      v23[1] = *(_QWORD *)(v5 + 8);
      v23[2] = *(_QWORD *)(v5 + 16);
      v23[3] = *(_QWORD *)(v5 + 24);
      v43 = v23[6];
      v23[4] = *(_QWORD *)(v5 + 32);
      v23[5] = *(_QWORD *)(v5 + 40);
      v23[6] = *(_QWORD *)(v5 + 48);
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      for ( *(_QWORD *)(v5 + 48) = 0; v25 != v26; v26 += 13 )
      {
        v27 = (_QWORD *)v26[9];
        if ( v27 != v26 + 11 )
          j_j___libc_free_0(v27, v26[11] + 1LL);
        v28 = v26[6];
        if ( v28 )
          j_j___libc_free_0(v28, v26[8] - v28);
      }
      if ( v46 )
        j_j___libc_free_0(v46, v43 - (_QWORD)v46);
      v29 = *(_DWORD *)(v5 + 56);
      v23 += 12;
      v5 += 96;
      *((_DWORD *)v23 - 10) = v29;
      *((_DWORD *)v23 - 9) = *(_DWORD *)(v5 - 36);
      *((_DWORD *)v23 - 8) = *(_DWORD *)(v5 - 32);
      *(v23 - 3) = *(_QWORD *)(v5 - 24);
      *((_BYTE *)v23 - 16) = *(_BYTE *)(v5 - 16);
      *((_BYTE *)v23 - 15) = *(_BYTE *)(v5 - 15);
      *((_DWORD *)v23 - 3) = *(_DWORD *)(v5 - 12);
      *((_BYTE *)v23 - 8) = *(_BYTE *)(v5 - 8);
      *((_BYTE *)v23 - 7) = *(_BYTE *)(v5 - 7);
      --v40;
    }
    while ( v40 );
    a5 += v38;
  }
  return a5;
}
