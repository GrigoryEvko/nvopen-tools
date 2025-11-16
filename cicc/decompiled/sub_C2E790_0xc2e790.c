// Function: sub_C2E790
// Address: 0xc2e790
//
__int64 __fastcall sub_C2E790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  char v12; // al
  __int64 v14; // rax
  char *v15; // r15
  char *v16; // r13
  char *k; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 *v20; // r13
  __int64 *v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  bool v25; // zf
  _QWORD *v26; // r15
  __int64 v27; // rdx
  _QWORD *v28; // r13
  _QWORD *j; // rdx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // r14
  __int64 *v33; // r13
  __int64 v34; // rdi
  __int64 v35; // rdi
  char *v36; // r15
  char *v37; // r13
  char *i; // rax
  __int64 v39; // rdi
  unsigned int v40; // ecx
  __int64 *v41; // r13
  __int64 *v42; // r14
  __int64 v43; // rdi
  __int64 v44; // [rsp+0h] [rbp-E0h]
  __int64 *v45; // [rsp+8h] [rbp-D8h]
  __int64 v46; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v47[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v48[3]; // [rsp+30h] [rbp-B0h] BYREF
  char *v49; // [rsp+48h] [rbp-98h]
  int v50; // [rsp+50h] [rbp-90h]
  _BYTE v51[32]; // [rsp+58h] [rbp-88h] BYREF
  __int64 *v52; // [rsp+78h] [rbp-68h]
  unsigned int v53; // [rsp+80h] [rbp-60h]
  _BYTE v54[24]; // [rsp+88h] [rbp-58h] BYREF
  char v55; // [rsp+A0h] [rbp-40h]

  v5 = a3;
  if ( (_DWORD)a2 == 2 )
  {
    v23 = sub_22077B0(304);
    v11 = v23;
    if ( v23 )
    {
      v24 = 2;
      v55 = 0;
      sub_C31010(v23, 2, a4, v5, v47);
      if ( v55 )
      {
        v36 = v49;
        v55 = 0;
        v37 = &v49[8 * v50];
        if ( v49 != v37 )
        {
          for ( i = v49; ; i = v49 )
          {
            v39 = *(_QWORD *)v36;
            v40 = (unsigned int)((v36 - i) >> 3) >> 7;
            v24 = 4096LL << v40;
            if ( v40 >= 0x1E )
              v24 = 0x40000000000LL;
            v36 += 8;
            sub_C7D6A0(v39, v24, 16);
            if ( v37 == v36 )
              break;
          }
        }
        v41 = v52;
        v42 = &v52[2 * v53];
        if ( v52 != v42 )
        {
          do
          {
            v24 = v41[1];
            v43 = *v41;
            v41 += 2;
            sub_C7D6A0(v43, v24, 16);
          }
          while ( v42 != v41 );
          v42 = v52;
        }
        if ( v42 != (__int64 *)v54 )
          _libc_free(v42, v24);
        if ( v49 != v51 )
          _libc_free(v49, v24);
        _libc_free(v47[0], v24);
      }
      *(_BYTE *)(v11 + 296) = 0;
      v25 = *(_BYTE *)(v11 + 160) == 0;
      *(_QWORD *)v11 = &unk_49DBE70;
      v45 = (__int64 *)(v11 + 136);
      v44 = v11 + 88;
      if ( !v25 )
      {
        v26 = *(_QWORD **)(v11 + 72);
        v27 = *(unsigned int *)(v11 + 80);
        *(_BYTE *)(v11 + 160) = 0;
        v28 = &v26[v27];
        if ( v26 != v28 )
        {
          for ( j = v26; ; j = *(_QWORD **)(v11 + 72) )
          {
            v30 = *v26;
            v31 = (unsigned int)(v26 - j) >> 7;
            v24 = 4096LL << v31;
            if ( v31 >= 0x1E )
              v24 = 0x40000000000LL;
            ++v26;
            sub_C7D6A0(v30, v24, 16);
            if ( v28 == v26 )
              break;
          }
        }
        v32 = *(__int64 **)(v11 + 120);
        v33 = &v32[2 * *(unsigned int *)(v11 + 128)];
        if ( v32 != v33 )
        {
          do
          {
            v24 = v32[1];
            v34 = *v32;
            v32 += 2;
            sub_C7D6A0(v34, v24, 16);
          }
          while ( v33 != v32 );
          v33 = *(__int64 **)(v11 + 120);
        }
        if ( v33 != v45 )
          _libc_free(v33, v24);
        v35 = *(_QWORD *)(v11 + 72);
        if ( v35 != v44 )
          _libc_free(v35, v24);
        _libc_free(*(_QWORD *)(v11 + 32), v24);
      }
      memset((void *)(v11 + 32), 0, 0x80u);
      *(_BYTE *)(v11 + 52) = 16;
      *(_QWORD *)(v11 + 144) = 1;
      *(_QWORD *)(v11 + 72) = v44;
      *(_QWORD *)(v11 + 80) = 0x400000000LL;
      *(_BYTE *)(v11 + 160) = 1;
      *(_QWORD *)(v11 + 120) = v45;
    }
    goto LABEL_10;
  }
  if ( (int)a2 > 2 )
  {
    if ( (_DWORD)a2 == 3 )
    {
      v10 = sub_22077B0(1984);
      v11 = v10;
      if ( v10 )
        sub_EFE770(v10, a4, v5);
      goto LABEL_10;
    }
LABEL_62:
    BUG();
  }
  if ( (_DWORD)a2 )
  {
    if ( (_DWORD)a2 == 1 )
    {
      v55 = 0;
      v14 = sub_22077B0(296);
      v11 = v14;
      if ( v14 )
      {
        a2 = a4;
        sub_C31760(v14, a4, v5, v47);
      }
      if ( v55 )
      {
        v15 = v49;
        v55 = 0;
        v16 = &v49[8 * v50];
        if ( v49 != v16 )
        {
          for ( k = v49; ; k = v49 )
          {
            v18 = *(_QWORD *)v15;
            v19 = (unsigned int)((v15 - k) >> 3) >> 7;
            a2 = 4096LL << v19;
            if ( v19 >= 0x1E )
              a2 = 0x40000000000LL;
            v15 += 8;
            sub_C7D6A0(v18, a2, 16);
            if ( v16 == v15 )
              break;
          }
        }
        v20 = v52;
        v21 = &v52[2 * v53];
        if ( v52 != v21 )
        {
          do
          {
            a2 = v20[1];
            v22 = *v20;
            v20 += 2;
            sub_C7D6A0(v22, a2, 16);
          }
          while ( v21 != v20 );
          v21 = v52;
        }
        if ( v21 != (__int64 *)v54 )
          _libc_free(v21, a2);
        if ( v49 != v51 )
          _libc_free(v49, a2);
        _libc_free(v47[0], a2);
      }
LABEL_10:
      v12 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = v11;
      *(_BYTE *)(a1 + 8) = v12 & 0xFC | 2;
      return a1;
    }
    goto LABEL_62;
  }
  v8 = sub_2241E50(a1, a2, a3, a4, a5);
  v47[0] = (__int64)v48;
  sub_C2E6E0(v47, "Unknown remark serializer format.", (__int64)"");
  sub_C63F00(&v46, v47, 22, v8);
  if ( (_QWORD *)v47[0] != v48 )
    j_j___libc_free_0(v47[0], v48[0] + 1LL);
  v9 = v46;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v9 & 0xFFFFFFFFFFFFFFFELL;
  return a1;
}
