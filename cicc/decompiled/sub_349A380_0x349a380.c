// Function: sub_349A380
// Address: 0x349a380
//
unsigned __int64 *__fastcall sub_349A380(unsigned __int64 *a1, int *a2, __int64 *a3)
{
  int *v4; // r13
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  int v10; // r10d
  unsigned __int64 v11; // r12
  int v12; // r9d
  int v13; // r8d
  __int64 v14; // rax
  _BYTE *v15; // r15
  unsigned __int64 v16; // r12
  __int64 v17; // r14
  unsigned __int64 v18; // r13
  unsigned __int64 *v19; // r14
  unsigned __int64 *v20; // r12
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // r12
  __int64 i; // rbx
  int v24; // eax
  int v25; // eax
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  __int64 *v28; // rdi
  int v29; // eax
  int v30; // edx
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rdi
  __int64 v33; // r14
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  unsigned __int64 v36; // r15
  unsigned __int64 *v37; // rbx
  unsigned __int64 *v38; // r14
  unsigned __int64 *v39; // r12
  unsigned __int64 v41; // rbx
  unsigned __int64 v43; // [rsp+28h] [rbp-118h]
  __int64 v44; // [rsp+30h] [rbp-110h]
  __int64 v45; // [rsp+38h] [rbp-108h]
  unsigned __int64 v46; // [rsp+40h] [rbp-100h]
  unsigned __int64 v47; // [rsp+48h] [rbp-F8h]
  __int64 v48; // [rsp+50h] [rbp-F0h]
  __int64 v49; // [rsp+58h] [rbp-E8h]
  unsigned __int64 *v50; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+68h] [rbp-D8h]
  _BYTE v52[32]; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v53; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v54; // [rsp+98h] [rbp-A8h]
  _BYTE v55[160]; // [rsp+A0h] [rbp-A0h] BYREF

  v4 = a2;
  v47 = a1[1];
  v46 = *a1;
  v5 = 0xEF7BDEF7BDEF7BDFLL * ((__int64)(v47 - *a1) >> 3);
  if ( v5 == 0x84210842108421LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xEF7BDEF7BDEF7BDFLL * ((__int64)(v47 - *a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x1084210842108421LL * ((__int64)(v47 - *a1) >> 3);
  if ( v7 )
  {
    v41 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v43 = 0;
      v9 = 248;
      v45 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x84210842108421LL )
      v8 = 0x84210842108421LL;
    v41 = 248 * v8;
  }
  v45 = sub_22077B0(v41);
  v43 = v45 + v41;
  v9 = v45 + 248;
LABEL_7:
  v10 = *((_DWORD *)a3 + 6);
  v11 = (unsigned __int64)a2 + v45 - v46;
  v48 = *a3;
  v49 = a3[1];
  v50 = (unsigned __int64 *)v52;
  v51 = 0x100000000LL;
  if ( v10 )
    sub_34996D0((__int64)&v50, (__int64)(a3 + 2));
  v12 = *((_DWORD *)a3 + 18);
  v53 = v55;
  v54 = 0x200000000LL;
  if ( v12 )
    sub_3499E80((__int64)&v53, (__int64)(a3 + 8));
  if ( v11 )
  {
    v13 = v51;
    *(_QWORD *)v11 = v48;
    *(_QWORD *)(v11 + 8) = v49;
    *(_QWORD *)(v11 + 16) = v11 + 32;
    *(_QWORD *)(v11 + 24) = 0x100000000LL;
    if ( v13 )
      sub_34996D0(v11 + 16, (__int64)&v50);
    *(_QWORD *)(v11 + 64) = v11 + 80;
    *(_QWORD *)(v11 + 72) = 0x200000000LL;
    v14 = (unsigned int)v54;
    if ( (_DWORD)v54 )
    {
      sub_3499E80(v11 + 64, (__int64)&v53);
      v14 = (unsigned int)v54;
    }
    *(_QWORD *)(v11 + 200) = 0;
    *(_QWORD *)(v11 + 192) = v11 + 208;
    *(_BYTE *)(v11 + 208) = 0;
    *(_DWORD *)(v11 + 224) = 6;
    *(_QWORD *)(v11 + 232) = 0;
    *(_WORD *)(v11 + 240) = 1;
  }
  else
  {
    v14 = (unsigned int)v54;
  }
  v15 = v53;
  v16 = (unsigned __int64)&v53[56 * v14];
  if ( v53 != (_BYTE *)v16 )
  {
    do
    {
      v17 = *(unsigned int *)(v16 - 40);
      v18 = *(_QWORD *)(v16 - 48);
      v16 -= 56LL;
      v19 = (unsigned __int64 *)(v18 + 32 * v17);
      if ( (unsigned __int64 *)v18 != v19 )
      {
        do
        {
          v19 -= 4;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            j_j___libc_free_0(*v19);
        }
        while ( (unsigned __int64 *)v18 != v19 );
        v18 = *(_QWORD *)(v16 + 8);
      }
      if ( v18 != v16 + 24 )
        _libc_free(v18);
    }
    while ( v15 != (_BYTE *)v16 );
    v4 = a2;
    v16 = (unsigned __int64)v53;
  }
  if ( (_BYTE *)v16 != v55 )
    _libc_free(v16);
  v20 = v50;
  v21 = &v50[4 * (unsigned int)v51];
  if ( v50 != v21 )
  {
    do
    {
      v21 -= 4;
      if ( (unsigned __int64 *)*v21 != v21 + 2 )
        j_j___libc_free_0(*v21);
    }
    while ( v20 != v21 );
    v21 = v50;
  }
  if ( v21 != (unsigned __int64 *)v52 )
    _libc_free((unsigned __int64)v21);
  v22 = v46;
  if ( a2 != (int *)v46 )
  {
    for ( i = v45; ; i += 248 )
    {
      if ( i )
      {
        *(_DWORD *)i = *(_DWORD *)v22;
        *(_DWORD *)(i + 4) = *(_DWORD *)(v22 + 4);
        *(_BYTE *)(i + 8) = *(_BYTE *)(v22 + 8);
        *(_BYTE *)(i + 9) = *(_BYTE *)(v22 + 9);
        *(_BYTE *)(i + 10) = *(_BYTE *)(v22 + 10);
        *(_BYTE *)(i + 11) = *(_BYTE *)(v22 + 11);
        v24 = *(_DWORD *)(v22 + 12);
        *(_DWORD *)(i + 24) = 0;
        *(_DWORD *)(i + 12) = v24;
        *(_QWORD *)(i + 16) = i + 32;
        *(_DWORD *)(i + 28) = 1;
        if ( *(_DWORD *)(v22 + 24) )
          sub_34994E0(i + 16, v22 + 16);
        *(_DWORD *)(i + 72) = 0;
        *(_QWORD *)(i + 64) = i + 80;
        *(_DWORD *)(i + 76) = 2;
        if ( *(_DWORD *)(v22 + 72) )
          sub_3499B90(i + 64, v22 + 64);
        *(_QWORD *)(i + 192) = i + 208;
        sub_343F8D0((__int64 *)(i + 192), *(_BYTE **)(v22 + 192), *(_QWORD *)(v22 + 192) + *(_QWORD *)(v22 + 200));
        *(_DWORD *)(i + 224) = *(_DWORD *)(v22 + 224);
        *(_QWORD *)(i + 232) = *(_QWORD *)(v22 + 232);
        *(_WORD *)(i + 240) = *(_WORD *)(v22 + 240);
      }
      v22 += 248LL;
      if ( a2 == (int *)v22 )
        break;
    }
    v9 = i + 496;
  }
  if ( a2 != (int *)v47 )
  {
    do
    {
      v29 = *v4;
      v30 = v4[6];
      *(_DWORD *)(v9 + 24) = 0;
      *(_DWORD *)(v9 + 28) = 1;
      *(_DWORD *)v9 = v29;
      *(_DWORD *)(v9 + 4) = v4[1];
      *(_BYTE *)(v9 + 8) = *((_BYTE *)v4 + 8);
      *(_BYTE *)(v9 + 9) = *((_BYTE *)v4 + 9);
      *(_BYTE *)(v9 + 10) = *((_BYTE *)v4 + 10);
      *(_BYTE *)(v9 + 11) = *((_BYTE *)v4 + 11);
      *(_DWORD *)(v9 + 12) = v4[3];
      *(_QWORD *)(v9 + 16) = v9 + 32;
      if ( v30 )
        sub_34994E0(v9 + 16, (__int64)(v4 + 4));
      *(_DWORD *)(v9 + 72) = 0;
      *(_QWORD *)(v9 + 64) = v9 + 80;
      v25 = v4[18];
      *(_DWORD *)(v9 + 76) = 2;
      if ( v25 )
        sub_3499B90(v9 + 64, (__int64)(v4 + 16));
      v26 = (_BYTE *)*((_QWORD *)v4 + 24);
      v27 = *((_QWORD *)v4 + 25);
      v28 = (__int64 *)(v9 + 192);
      *(_QWORD *)(v9 + 192) = v9 + 208;
      v4 += 62;
      v9 += 248;
      sub_343F8D0(v28, v26, (__int64)&v26[v27]);
      *(_DWORD *)(v9 - 24) = *(v4 - 6);
      *(_QWORD *)(v9 - 16) = *((_QWORD *)v4 - 2);
      *(_WORD *)(v9 - 8) = *((_WORD *)v4 - 4);
    }
    while ( (int *)v47 != v4 );
  }
  v31 = v46;
  if ( v46 != v47 )
  {
    v44 = v9;
    do
    {
      v32 = *(_QWORD *)(v31 + 192);
      if ( v32 != v31 + 208 )
        j_j___libc_free_0(v32);
      v33 = *(_QWORD *)(v31 + 64);
      v34 = v33 + 56LL * *(unsigned int *)(v31 + 72);
      if ( v33 != v34 )
      {
        do
        {
          v35 = *(unsigned int *)(v34 - 40);
          v36 = *(_QWORD *)(v34 - 48);
          v34 -= 56LL;
          v35 *= 32;
          v37 = (unsigned __int64 *)(v36 + v35);
          if ( v36 != v36 + v35 )
          {
            do
            {
              v37 -= 4;
              if ( (unsigned __int64 *)*v37 != v37 + 2 )
                j_j___libc_free_0(*v37);
            }
            while ( (unsigned __int64 *)v36 != v37 );
            v36 = *(_QWORD *)(v34 + 8);
          }
          if ( v36 != v34 + 24 )
            _libc_free(v36);
        }
        while ( v33 != v34 );
        v34 = *(_QWORD *)(v31 + 64);
      }
      if ( v34 != v31 + 80 )
        _libc_free(v34);
      v38 = *(unsigned __int64 **)(v31 + 16);
      v39 = &v38[4 * *(unsigned int *)(v31 + 24)];
      if ( v38 != v39 )
      {
        do
        {
          v39 -= 4;
          if ( (unsigned __int64 *)*v39 != v39 + 2 )
            j_j___libc_free_0(*v39);
        }
        while ( v38 != v39 );
        v39 = *(unsigned __int64 **)(v31 + 16);
      }
      if ( v39 != (unsigned __int64 *)(v31 + 32) )
        _libc_free((unsigned __int64)v39);
      v31 += 248LL;
    }
    while ( v31 != v47 );
    v9 = v44;
  }
  if ( v46 )
    j_j___libc_free_0(v46);
  *a1 = v45;
  a1[1] = v9;
  a1[2] = v43;
  return a1;
}
