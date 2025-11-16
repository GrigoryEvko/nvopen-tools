// Function: sub_20C00F0
// Address: 0x20c00f0
//
__int64 *__fastcall sub_20C00F0(__int64 *a1, int *a2, __int64 *a3)
{
  int *v4; // r13
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // r8
  int v15; // edi
  __int64 v16; // rax
  _BYTE *v17; // r15
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // r14
  unsigned __int64 v21; // r13
  _QWORD *v22; // r14
  _QWORD *v23; // r12
  _QWORD *v24; // r14
  __int64 v25; // r12
  __int64 i; // rbx
  __int64 v27; // rcx
  int v28; // eax
  __int64 v29; // rcx
  int v30; // eax
  _BYTE *v31; // rsi
  __int64 v32; // rdx
  __int64 *v33; // rdi
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // rdi
  __int64 v38; // r14
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  unsigned __int64 v41; // r15
  _QWORD *v42; // rbx
  _QWORD *v43; // r14
  _QWORD *v44; // r12
  __int64 v46; // rbx
  __int64 v48; // [rsp+28h] [rbp-118h]
  __int64 v49; // [rsp+30h] [rbp-110h]
  __int64 v50; // [rsp+38h] [rbp-108h]
  __int64 v51; // [rsp+40h] [rbp-100h]
  __int64 v52; // [rsp+48h] [rbp-F8h]
  __int64 v53; // [rsp+50h] [rbp-F0h]
  __int64 v54; // [rsp+58h] [rbp-E8h]
  _BYTE *v55; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v56; // [rsp+68h] [rbp-D8h]
  _BYTE v57[32]; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v58; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+98h] [rbp-A8h]
  _BYTE v60[160]; // [rsp+A0h] [rbp-A0h] BYREF

  v4 = a2;
  v5 = *a1;
  v52 = a1[1];
  v51 = *a1;
  v6 = 0xEF7BDEF7BDEF7BDFLL * ((v52 - *a1) >> 3);
  if ( v6 == 0x84210842108421LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xEF7BDEF7BDEF7BDFLL * ((v52 - *a1) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 - 0x1084210842108421LL * ((v52 - *a1) >> 3);
  v10 = v8;
  if ( v8 )
  {
    v46 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v48 = 0;
      v11 = 248;
      v50 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x84210842108421LL )
      v9 = 0x84210842108421LL;
    v46 = 248 * v9;
  }
  v50 = sub_22077B0(v46);
  v48 = v50 + v46;
  v11 = v50 + 248;
LABEL_7:
  v12 = *((unsigned int *)a3 + 6);
  v13 = (__int64)a2 + v50 - v51;
  v53 = *a3;
  v54 = a3[1];
  v55 = v57;
  v56 = 0x100000000LL;
  if ( (_DWORD)v12 )
    sub_20BDAA0((__int64)&v55, (__int64)(a3 + 2), v10, v5);
  v14 = *((unsigned int *)a3 + 18);
  v58 = v60;
  v59 = 0x200000000LL;
  if ( (_DWORD)v14 )
    sub_20BFC20((__int64)&v58, (__int64)(a3 + 8));
  if ( v13 )
  {
    v15 = v56;
    *(_QWORD *)v13 = v53;
    *(_QWORD *)(v13 + 8) = v54;
    *(_QWORD *)(v13 + 16) = v13 + 32;
    *(_QWORD *)(v13 + 24) = 0x100000000LL;
    if ( v15 )
      sub_20BDAA0(v13 + 16, (__int64)&v55, v10, v5);
    *(_QWORD *)(v13 + 64) = v13 + 80;
    *(_QWORD *)(v13 + 72) = 0x200000000LL;
    v16 = (unsigned int)v59;
    if ( (_DWORD)v59 )
    {
      sub_20BFC20(v13 + 64, (__int64)&v58);
      v16 = (unsigned int)v59;
    }
    *(_QWORD *)(v13 + 200) = 0;
    *(_QWORD *)(v13 + 192) = v13 + 208;
    *(_BYTE *)(v13 + 208) = 0;
    *(_DWORD *)(v13 + 224) = 4;
    *(_QWORD *)(v13 + 232) = 0;
    *(_BYTE *)(v13 + 240) = 1;
  }
  else
  {
    v16 = (unsigned int)v59;
  }
  v17 = v58;
  v18 = 7 * v16;
  v19 = (unsigned __int64)&v58[56 * v16];
  if ( v58 != (_BYTE *)v19 )
  {
    do
    {
      v20 = *(unsigned int *)(v19 - 40);
      v21 = *(_QWORD *)(v19 - 48);
      v19 -= 56LL;
      v22 = (_QWORD *)(v21 + 32 * v20);
      if ( (_QWORD *)v21 != v22 )
      {
        do
        {
          v22 -= 4;
          if ( (_QWORD *)*v22 != v22 + 2 )
            j_j___libc_free_0(*v22, v22[2] + 1LL);
        }
        while ( (_QWORD *)v21 != v22 );
        v21 = *(_QWORD *)(v19 + 8);
      }
      if ( v21 != v19 + 24 )
        _libc_free(v21);
    }
    while ( v17 != (_BYTE *)v19 );
    v4 = a2;
    v19 = (unsigned __int64)v58;
  }
  if ( (_BYTE *)v19 != v60 )
    _libc_free(v19);
  v23 = v55;
  v24 = &v55[32 * (unsigned int)v56];
  if ( v55 != (_BYTE *)v24 )
  {
    do
    {
      v24 -= 4;
      if ( (_QWORD *)*v24 != v24 + 2 )
        j_j___libc_free_0(*v24, v24[2] + 1LL);
    }
    while ( v23 != v24 );
    v24 = v55;
  }
  if ( v24 != (_QWORD *)v57 )
    _libc_free((unsigned __int64)v24);
  v25 = v51;
  if ( a2 != (int *)v51 )
  {
    for ( i = v50; ; i += 248 )
    {
      if ( i )
      {
        *(_DWORD *)i = *(_DWORD *)v25;
        *(_DWORD *)(i + 4) = *(_DWORD *)(v25 + 4);
        *(_BYTE *)(i + 8) = *(_BYTE *)(v25 + 8);
        *(_BYTE *)(i + 9) = *(_BYTE *)(v25 + 9);
        *(_BYTE *)(i + 10) = *(_BYTE *)(v25 + 10);
        *(_BYTE *)(i + 11) = *(_BYTE *)(v25 + 11);
        v28 = *(_DWORD *)(v25 + 12);
        *(_DWORD *)(i + 24) = 0;
        *(_DWORD *)(i + 12) = v28;
        *(_QWORD *)(i + 16) = i + 32;
        *(_DWORD *)(i + 28) = 1;
        if ( *(_DWORD *)(v25 + 24) )
          sub_20BD8D0(i + 16, (__int64 *)(v25 + 16), v18, v5, v14, v12);
        *(_DWORD *)(i + 72) = 0;
        *(_QWORD *)(i + 64) = i + 80;
        *(_DWORD *)(i + 76) = 2;
        v27 = *(unsigned int *)(v25 + 72);
        if ( (_DWORD)v27 )
          sub_20BF940(i + 64, v25 + 64, v18, v27, v14, v12);
        *(_QWORD *)(i + 192) = i + 208;
        sub_20A0C10((__int64 *)(i + 192), *(_BYTE **)(v25 + 192), *(_QWORD *)(v25 + 192) + *(_QWORD *)(v25 + 200));
        *(_DWORD *)(i + 224) = *(_DWORD *)(v25 + 224);
        *(_QWORD *)(i + 232) = *(_QWORD *)(v25 + 232);
        *(_BYTE *)(i + 240) = *(_BYTE *)(v25 + 240);
      }
      v25 += 248;
      if ( a2 == (int *)v25 )
        break;
    }
    v11 = i + 496;
  }
  v29 = v52;
  if ( a2 != (int *)v52 )
  {
    do
    {
      v34 = *v4;
      v35 = (unsigned int)v4[6];
      *(_DWORD *)(v11 + 24) = 0;
      *(_DWORD *)(v11 + 28) = 1;
      *(_DWORD *)v11 = v34;
      *(_DWORD *)(v11 + 4) = v4[1];
      *(_BYTE *)(v11 + 8) = *((_BYTE *)v4 + 8);
      *(_BYTE *)(v11 + 9) = *((_BYTE *)v4 + 9);
      *(_BYTE *)(v11 + 10) = *((_BYTE *)v4 + 10);
      *(_BYTE *)(v11 + 11) = *((_BYTE *)v4 + 11);
      *(_DWORD *)(v11 + 12) = v4[3];
      *(_QWORD *)(v11 + 16) = v11 + 32;
      if ( (_DWORD)v35 )
        sub_20BD8D0(v11 + 16, (__int64 *)v4 + 2, v35, v29, v14, v12);
      *(_DWORD *)(v11 + 72) = 0;
      *(_QWORD *)(v11 + 64) = v11 + 80;
      v30 = v4[18];
      *(_DWORD *)(v11 + 76) = 2;
      if ( v30 )
        sub_20BF940(v11 + 64, (__int64)(v4 + 16), v35, v29, v14, v12);
      v31 = (_BYTE *)*((_QWORD *)v4 + 24);
      v32 = *((_QWORD *)v4 + 25);
      v33 = (__int64 *)(v11 + 192);
      *(_QWORD *)(v11 + 192) = v11 + 208;
      v4 += 62;
      v11 += 248;
      sub_20A0C10(v33, v31, (__int64)&v31[v32]);
      *(_DWORD *)(v11 - 24) = *(v4 - 6);
      *(_QWORD *)(v11 - 16) = *((_QWORD *)v4 - 2);
      *(_BYTE *)(v11 - 8) = *((_BYTE *)v4 - 8);
    }
    while ( (int *)v52 != v4 );
  }
  v36 = v51;
  if ( v51 != v52 )
  {
    v49 = v11;
    do
    {
      v37 = *(_QWORD *)(v36 + 192);
      if ( v37 != v36 + 208 )
        j_j___libc_free_0(v37, *(_QWORD *)(v36 + 208) + 1LL);
      v38 = *(_QWORD *)(v36 + 64);
      v39 = v38 + 56LL * *(unsigned int *)(v36 + 72);
      if ( v38 != v39 )
      {
        do
        {
          v40 = *(unsigned int *)(v39 - 40);
          v41 = *(_QWORD *)(v39 - 48);
          v39 -= 56LL;
          v40 *= 32;
          v42 = (_QWORD *)(v41 + v40);
          if ( v41 != v41 + v40 )
          {
            do
            {
              v42 -= 4;
              if ( (_QWORD *)*v42 != v42 + 2 )
                j_j___libc_free_0(*v42, v42[2] + 1LL);
            }
            while ( (_QWORD *)v41 != v42 );
            v41 = *(_QWORD *)(v39 + 8);
          }
          if ( v41 != v39 + 24 )
            _libc_free(v41);
        }
        while ( v38 != v39 );
        v39 = *(_QWORD *)(v36 + 64);
      }
      if ( v39 != v36 + 80 )
        _libc_free(v39);
      v43 = *(_QWORD **)(v36 + 16);
      v44 = &v43[4 * *(unsigned int *)(v36 + 24)];
      if ( v43 != v44 )
      {
        do
        {
          v44 -= 4;
          if ( (_QWORD *)*v44 != v44 + 2 )
            j_j___libc_free_0(*v44, v44[2] + 1LL);
        }
        while ( v43 != v44 );
        v44 = *(_QWORD **)(v36 + 16);
      }
      if ( v44 != (_QWORD *)(v36 + 32) )
        _libc_free((unsigned __int64)v44);
      v36 += 248;
    }
    while ( v36 != v52 );
    v11 = v49;
  }
  if ( v51 )
    j_j___libc_free_0(v51, a1[2] - v51);
  *a1 = v50;
  a1[1] = v11;
  a1[2] = v48;
  return a1;
}
