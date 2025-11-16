// Function: sub_1495470
// Address: 0x1495470
//
__int64 *__fastcall sub_1495470(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  int v10; // r10d
  __int64 *v11; // rbx
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rax
  _BYTE *v15; // rbx
  _BYTE *v16; // r12
  __int64 v17; // r15
  __int64 v18; // rdx
  _QWORD *v19; // r13
  _QWORD *v20; // r14
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rcx
  int v25; // r8d
  unsigned int v26; // edx
  __int64 *v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 *v30; // r12
  __int64 v31; // r15
  __int64 v32; // rax
  _QWORD *v33; // rbx
  _QWORD *v34; // r13
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  __int64 *v39; // [rsp+28h] [rbp-C8h]
  _QWORD v40[3]; // [rsp+38h] [rbp-B8h] BYREF
  char v41; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+68h] [rbp-88h]
  char v43; // [rsp+70h] [rbp-80h]
  __int64 *v44; // [rsp+80h] [rbp-70h] BYREF
  _BYTE *v45; // [rsp+88h] [rbp-68h] BYREF
  __int64 v46; // [rsp+90h] [rbp-60h] BYREF
  _BYTE v47[16]; // [rsp+98h] [rbp-58h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-48h]
  __int64 v49; // [rsp+B0h] [rbp-40h]
  char v50; // [rsp+B8h] [rbp-38h]

  v2 = a1 + 560;
  v44 = (__int64 *)a2;
  v45 = v47;
  v4 = *(_DWORD *)(a1 + 584);
  v40[1] = &v41;
  v40[2] = 0x100000000LL;
  v42 = 0;
  v43 = 0;
  v46 = 0x100000000LL;
  v49 = 0;
  v50 = 0;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 560);
LABEL_51:
    sub_1469520(v2, 2 * v4);
    goto LABEL_52;
  }
  v5 = *(_QWORD *)(a1 + 568);
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v5 + ((unsigned __int64)v6 << 6));
  v8 = *v7;
  if ( a2 == *v7 )
    return v7 + 1;
  v10 = 1;
  v11 = 0;
  while ( v8 != -8 )
  {
    if ( !v11 && v8 == -16 )
      v11 = v7;
    v6 = (v4 - 1) & (v10 + v6);
    v7 = (__int64 *)(v5 + ((unsigned __int64)v6 << 6));
    v8 = *v7;
    if ( a2 == *v7 )
      return v7 + 1;
    ++v10;
  }
  if ( !v11 )
    v11 = v7;
  ++*(_QWORD *)(a1 + 560);
  v12 = *(_DWORD *)(a1 + 576) + 1;
  if ( 4 * v12 >= 3 * v4 )
    goto LABEL_51;
  v13 = a2;
  if ( v4 - *(_DWORD *)(a1 + 580) - v12 > v4 >> 3 )
    goto LABEL_11;
  sub_1469520(v2, v4);
LABEL_52:
  sub_145FC70(v2, (__int64 *)&v44, v40);
  v11 = (__int64 *)v40[0];
  v13 = (__int64)v44;
  v12 = *(_DWORD *)(a1 + 576) + 1;
LABEL_11:
  *(_DWORD *)(a1 + 576) = v12;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 580);
  *v11 = v13;
  v11[1] = (__int64)(v11 + 3);
  v11[2] = 0x100000000LL;
  if ( (_DWORD)v46 )
  {
    sub_145E880((__int64)(v11 + 1), (__int64)&v45);
    v14 = (unsigned int)v46;
    v11[6] = v49;
    *((_BYTE *)v11 + 56) = v50;
    v15 = v45;
    v16 = &v45[24 * v14];
    if ( v45 == v16 )
      goto LABEL_29;
    do
    {
      v17 = *((_QWORD *)v16 - 1);
      v16 -= 24;
      if ( v17 )
      {
        v18 = *(unsigned int *)(v17 + 208);
        *(_QWORD *)v17 = &unk_49EC708;
        if ( (_DWORD)v18 )
        {
          v19 = *(_QWORD **)(v17 + 192);
          v20 = &v19[7 * v18];
          do
          {
            if ( *v19 != -16 && *v19 != -8 )
            {
              v21 = v19[1];
              if ( (_QWORD *)v21 != v19 + 3 )
                _libc_free(v21);
            }
            v19 += 7;
          }
          while ( v20 != v19 );
        }
        j___libc_free_0(*(_QWORD *)(v17 + 192));
        v22 = *(_QWORD *)(v17 + 40);
        if ( v22 != v17 + 56 )
          _libc_free(v22);
        j_j___libc_free_0(v17, 216);
      }
    }
    while ( v15 != v16 );
  }
  else
  {
    v11[6] = v49;
    *((_BYTE *)v11 + 56) = v50;
  }
  v16 = v45;
LABEL_29:
  if ( v16 != v47 )
    _libc_free((unsigned __int64)v16);
  sub_1473410((__int64)&v44, a1, a2, 1u);
  v23 = *(unsigned int *)(a1 + 584);
  v24 = *(_QWORD *)(a1 + 568);
  if ( !(_DWORD)v23 )
  {
LABEL_49:
    v27 = (__int64 *)(v24 + (v23 << 6));
    goto LABEL_33;
  }
  v25 = 1;
  v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v27 = (__int64 *)(v24 + ((unsigned __int64)v26 << 6));
  v28 = *v27;
  if ( a2 != *v27 )
  {
    while ( v28 != -8 )
    {
      v26 = (v23 - 1) & (v25 + v26);
      v27 = (__int64 *)(v24 + ((unsigned __int64)v26 << 6));
      v28 = *v27;
      if ( a2 == *v27 )
        goto LABEL_33;
      ++v25;
    }
    goto LABEL_49;
  }
LABEL_33:
  v39 = v27 + 1;
  sub_145E880((__int64)(v27 + 1), (__int64)&v44);
  v27[6] = v48;
  *((_BYTE *)v27 + 56) = v49;
  v29 = (__int64)v44;
  v30 = &v44[3 * (unsigned int)v45];
  if ( v44 != v30 )
  {
    do
    {
      v31 = *(v30 - 1);
      v30 -= 3;
      if ( v31 )
      {
        *(_QWORD *)v31 = &unk_49EC708;
        v32 = *(unsigned int *)(v31 + 208);
        if ( (_DWORD)v32 )
        {
          v33 = *(_QWORD **)(v31 + 192);
          v34 = &v33[7 * v32];
          do
          {
            if ( *v33 != -16 && *v33 != -8 )
            {
              v35 = v33[1];
              if ( (_QWORD *)v35 != v33 + 3 )
                _libc_free(v35);
            }
            v33 += 7;
          }
          while ( v34 != v33 );
        }
        j___libc_free_0(*(_QWORD *)(v31 + 192));
        v36 = *(_QWORD *)(v31 + 40);
        if ( v36 != v31 + 56 )
          _libc_free(v36);
        j_j___libc_free_0(v31, 216);
      }
    }
    while ( (__int64 *)v29 != v30 );
    v30 = v44;
  }
  if ( v30 != &v46 )
    _libc_free((unsigned __int64)v30);
  return v39;
}
