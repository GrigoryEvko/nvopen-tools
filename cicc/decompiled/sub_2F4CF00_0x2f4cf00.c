// Function: sub_2F4CF00
// Address: 0x2f4cf00
//
void __fastcall sub_2F4CF00(unsigned __int64 a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r9
  int v14; // r13d
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 *k; // rax
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 v21; // rsi
  __int64 *v22; // rbx
  unsigned __int64 v23; // r13
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 j; // rcx
  _QWORD *v28; // rbx
  __int64 v29; // r14
  __int64 v30; // r13
  _BYTE *v31; // rax
  int v32; // r13d
  _BYTE *v33; // rdx
  __int64 v34; // rax
  __int64 i; // r14
  __int64 v36; // rdx
  __int64 v37; // r12
  __int64 v38; // r15
  __int64 v39; // r8
  int v40; // eax
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // r12
  int v43; // edi
  _BYTE *v44; // rsi
  __int64 v45; // r13
  __int64 v46; // r14
  __int64 v47; // rsi
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 *v51; // rcx
  __int64 v52; // rsi
  int v53; // edi
  _BYTE *v54; // rsi
  unsigned __int64 v55; // [rsp+0h] [rbp-C0h]
  __int64 v56; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v57; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v58; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+28h] [rbp-98h]
  unsigned __int64 v60; // [rsp+28h] [rbp-98h]
  unsigned __int64 v61; // [rsp+28h] [rbp-98h]
  _BYTE *v62; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v63; // [rsp+38h] [rbp-88h]
  _BYTE v64[32]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v65; // [rsp+60h] [rbp-60h] BYREF
  __int64 v66; // [rsp+68h] [rbp-58h]
  _BYTE v67[80]; // [rsp+70h] [rbp-50h] BYREF

  v1 = a1 - 288;
  v2 = a1 + 1136;
  v3 = a1;
  do
  {
    v4 = *(_QWORD *)(v2 + 184);
    if ( v4 != v2 + 200 )
      _libc_free(v4);
    v5 = *(_QWORD *)(v2 + 144);
    if ( v5 != v2 + 160 )
      _libc_free(v5);
    v6 = *(unsigned int *)(v2 + 136);
    if ( (_DWORD)v6 )
    {
      v7 = *(_QWORD **)(v2 + 120);
      v8 = &v7[19 * v6];
      do
      {
        if ( *v7 != -4096 && *v7 != -8192 )
        {
          v9 = v7[10];
          if ( (_QWORD *)v9 != v7 + 12 )
            _libc_free(v9);
          v10 = v7[1];
          if ( (_QWORD *)v10 != v7 + 3 )
            _libc_free(v10);
        }
        v7 += 19;
      }
      while ( v8 != v7 );
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 120), 152LL * *(unsigned int *)(v2 + 136), 8);
    v11 = *(_QWORD *)(v2 + 40);
    if ( v11 != v2 + 56 )
      _libc_free(v11);
    v2 -= 712LL;
  }
  while ( v1 != v2 );
  sub_C7D6A0(*(_QWORD *)(v3 + 400), 16LL * *(unsigned int *)(v3 + 416), 8);
  v14 = *(_DWORD *)(v3 + 376);
  if ( v14 )
  {
    LODWORD(j) = *(_DWORD *)(v3 + 380);
    v28 = (_QWORD *)(v3 + 192);
    v62 = v64;
    v63 = 0x400000000LL;
    v65 = v67;
    v66 = 0x400000000LL;
    if ( (_DWORD)j )
    {
      v29 = *(_QWORD *)(v3 + 200);
      v30 = 1;
      j = 0;
      v31 = v64;
      while ( 1 )
      {
        *(_QWORD *)&v31[8 * j] = v29;
        j = (unsigned int)(v63 + 1);
        LODWORD(v63) = v63 + 1;
        if ( *(_DWORD *)(v3 + 380) == (_DWORD)v30 )
          break;
        v29 = *(_QWORD *)(v3 + 8 * v30 + 200);
        if ( j + 1 > (unsigned __int64)HIDWORD(v63) )
        {
          sub_C8D5F0((__int64)&v62, v64, j + 1, 8u, v12, v13);
          j = (unsigned int)v63;
        }
        v31 = v62;
        ++v30;
      }
      v32 = *(_DWORD *)(v3 + 376) - 1;
      if ( *(_DWORD *)(v3 + 376) != 1 )
      {
        v33 = v62;
        v34 = (unsigned int)v66;
LABEL_41:
        v55 = v3;
        if ( !(_DWORD)j )
          goto LABEL_56;
LABEL_42:
        v56 = 8LL * (unsigned int)(j - 1);
        for ( i = 0; ; i += 8 )
        {
          v36 = *(_QWORD *)&v33[i];
          v37 = 0;
          v38 = 8 * (v36 & 0x3F) + 8;
          while ( 1 )
          {
            v39 = *(_QWORD *)((v36 & 0xFFFFFFFFFFFFFFC0LL) + v37);
            if ( v34 + 1 > (unsigned __int64)HIDWORD(v66) )
            {
              v59 = *(_QWORD *)((v36 & 0xFFFFFFFFFFFFFFC0LL) + v37);
              sub_C8D5F0((__int64)&v65, v67, v34 + 1, 8u, v39, v13);
              v34 = (unsigned int)v66;
              v39 = v59;
            }
            v37 += 8;
            *(_QWORD *)&v65[8 * v34] = v39;
            v34 = (unsigned int)(v66 + 1);
            LODWORD(v66) = v66 + 1;
            if ( v38 == v37 )
              break;
            v36 = *(_QWORD *)&v62[i];
          }
          sub_2F4C150((__int64)v28, *(_QWORD *)&v62[i]);
          v33 = v62;
          if ( v56 == i )
            break;
          v34 = (unsigned int)v66;
        }
        for ( j = (unsigned int)v66; ; j = (unsigned int)v34 )
        {
          LODWORD(v63) = 0;
          v40 = HIDWORD(v63);
          if ( v33 != v64 )
          {
            v41 = (unsigned __int64)v65;
            if ( v65 != v67 )
              break;
          }
          v13 = (unsigned int)j;
          if ( HIDWORD(v63) < (unsigned int)j )
          {
            sub_C8D5F0((__int64)&v62, v64, (unsigned int)j, 8u, v12, (unsigned int)j);
            v12 = (unsigned int)v63;
            j = (unsigned int)v63;
            if ( HIDWORD(v66) < (unsigned int)v63 )
            {
              sub_C8D5F0((__int64)&v65, v67, (unsigned int)v63, 8u, (unsigned int)v63, v48);
              v12 = (unsigned int)v63;
              j = (unsigned int)v63;
            }
            v13 = (unsigned int)v66;
            v42 = (unsigned int)v66;
            if ( v12 <= (unsigned int)v66 )
              v42 = v12;
            if ( v42 )
            {
              v49 = 0;
              do
              {
                v50 = (__int64 *)&v65[v49];
                v51 = (__int64 *)&v62[v49];
                v49 += 8;
                v52 = *v51;
                *v51 = *v50;
                *v50 = v52;
              }
              while ( 8 * v42 != v49 );
              v12 = (unsigned int)v63;
              v13 = (unsigned int)v66;
              j = (unsigned int)v63;
            }
            if ( v12 > v13 )
            {
              v53 = v13;
              v54 = &v62[8 * v42];
              if ( v54 != &v62[8 * v12] )
              {
                v58 = v12;
                v61 = v13;
                memcpy(&v65[8 * v13], v54, 8 * v12 - 8 * v42);
                v53 = v66;
                v12 = v58;
                v13 = v61;
              }
              v12 -= v13;
              LODWORD(v63) = v42;
              j = (unsigned int)v42;
              LODWORD(v66) = v12 + v53;
LABEL_54:
              if ( !--v32 )
                goto LABEL_63;
              goto LABEL_55;
            }
          }
          else
          {
            v12 = 0;
            j = 0;
            v42 = 0;
          }
          if ( v13 <= v12 )
            goto LABEL_54;
          v43 = v12;
          v44 = &v65[8 * v42];
          if ( v44 != &v65[8 * v13] )
          {
            v57 = v13;
            v60 = v12;
            memcpy(&v62[8 * v12], v44, 8 * v13 - 8 * v42);
            v43 = v63;
            v13 = v57;
            v12 = v60;
          }
          v13 -= v12;
          LODWORD(v66) = v42;
          LODWORD(v63) = v13 + v43;
          j = (unsigned int)(v13 + v43);
          if ( !--v32 )
          {
LABEL_63:
            v3 = v55;
            goto LABEL_64;
          }
LABEL_55:
          v33 = v62;
          v34 = (unsigned int)v66;
          if ( (_DWORD)j )
            goto LABEL_42;
LABEL_56:
          ;
        }
        v65 = v33;
        v62 = (_BYTE *)v41;
        v63 = __PAIR64__(HIDWORD(v66), j);
        LODWORD(v66) = 0;
        HIDWORD(v66) = v40;
        goto LABEL_54;
      }
LABEL_64:
      if ( (_DWORD)j )
      {
        v45 = 8 * j;
        v46 = 0;
        do
        {
          v47 = *(_QWORD *)&v62[v46];
          v46 += 8;
          sub_2F4C150((__int64)v28, v47);
        }
        while ( v45 != v46 );
      }
      if ( v65 != v67 )
        _libc_free((unsigned __int64)v65);
    }
    else
    {
      v32 = v14 - 1;
      if ( v32 )
      {
        v33 = v64;
        v34 = 0;
        goto LABEL_41;
      }
    }
    if ( v62 != v64 )
      _libc_free((unsigned __int64)v62);
    do
    {
      *v28 = 0;
      v28 += 2;
      *(v28 - 1) = 0;
    }
    while ( (_QWORD *)(v3 + 336) != v28 );
  }
  v15 = *(__int64 **)(v3 + 112);
  v16 = *(unsigned int *)(v3 + 120);
  *(_QWORD *)(v3 + 88) = 0;
  v17 = &v15[v16];
  if ( v15 != v17 )
  {
    for ( k = v15; ; k = *(__int64 **)(v3 + 112) )
    {
      v19 = *v15;
      v20 = (unsigned int)(v15 - k) >> 7;
      v21 = 4096LL << v20;
      if ( v20 >= 0x1E )
        v21 = 0x40000000000LL;
      ++v15;
      sub_C7D6A0(v19, v21, 16);
      if ( v17 == v15 )
        break;
    }
  }
  v22 = *(__int64 **)(v3 + 160);
  v23 = (unsigned __int64)&v22[2 * *(unsigned int *)(v3 + 168)];
  if ( v22 != (__int64 *)v23 )
  {
    do
    {
      v24 = v22[1];
      v25 = *v22;
      v22 += 2;
      sub_C7D6A0(v25, v24, 16);
    }
    while ( (__int64 *)v23 != v22 );
    v23 = *(_QWORD *)(v3 + 160);
  }
  if ( v23 != v3 + 176 )
    _libc_free(v23);
  v26 = *(_QWORD *)(v3 + 112);
  if ( v26 != v3 + 128 )
    _libc_free(v26);
  j_j___libc_free_0(v3);
}
