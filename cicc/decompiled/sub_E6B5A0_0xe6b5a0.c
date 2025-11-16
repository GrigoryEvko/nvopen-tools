// Function: sub_E6B5A0
// Address: 0xe6b5a0
//
__int64 __fastcall sub_E6B5A0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  size_t v4; // rbx
  char *v5; // r13
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // rdx
  const char *v12; // rsi
  size_t v13; // r10
  size_t v14; // rdx
  unsigned __int64 v15; // rax
  char *v16; // rsi
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  bool v20; // zf
  char *v21; // rdi
  char *v22; // r9
  char *v23; // rsi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 v26; // rdx
  size_t v27; // r15
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  char *v32; // [rsp+8h] [rbp-1E8h]
  char *v33; // [rsp+8h] [rbp-1E8h]
  bool v34; // [rsp+10h] [rbp-1E0h]
  size_t v35; // [rsp+10h] [rbp-1E0h]
  size_t v36; // [rsp+10h] [rbp-1E0h]
  size_t v37; // [rsp+10h] [rbp-1E0h]
  size_t v38; // [rsp+10h] [rbp-1E0h]
  char v40; // [rsp+3Fh] [rbp-1B1h] BYREF
  char *v41; // [rsp+40h] [rbp-1B0h] BYREF
  unsigned __int64 v42; // [rsp+48h] [rbp-1A8h]
  __int64 v43; // [rsp+50h] [rbp-1A0h]
  __int64 v44; // [rsp+58h] [rbp-198h]
  __int64 v45; // [rsp+60h] [rbp-190h]
  __int64 v46; // [rsp+68h] [rbp-188h]
  const char **v47; // [rsp+70h] [rbp-180h]
  void *src; // [rsp+80h] [rbp-170h] BYREF
  size_t n; // [rsp+88h] [rbp-168h]
  __int64 v50; // [rsp+90h] [rbp-160h]
  _BYTE v51[136]; // [rsp+98h] [rbp-158h] BYREF
  const char *v52; // [rsp+120h] [rbp-D0h] BYREF
  unsigned __int64 v53; // [rsp+128h] [rbp-C8h]
  unsigned __int64 v54; // [rsp+130h] [rbp-C0h]
  _QWORD v55[23]; // [rsp+138h] [rbp-B8h] BYREF

  if ( !a2 )
  {
    v29 = sub_EA1530(80, 0, a1);
    v9 = v29;
    if ( !v29 )
      return v9;
    *(_QWORD *)(v29 + 24) = 0;
    *(_QWORD *)v29 = 0;
    *(_DWORD *)(v29 + 16) = 0;
    v30 = 2LL * a3;
    BYTE1(v30) |= 6u;
    *(_QWORD *)(v9 + 8) = *(_QWORD *)(v9 + 8) & 0xFFFF0000FFF00000LL | v30;
    goto LABEL_7;
  }
  v4 = *(_QWORD *)a2;
  v5 = (char *)(a2 + 24);
  if ( *(_QWORD *)a2 > 0xAu )
  {
    if ( *(_QWORD *)(a2 + 24) == 0x656D616E65525F2ELL && *(_WORD *)(a2 + 32) == 11876 && *(_BYTE *)(a2 + 34) == 46 )
      goto LABEL_12;
  }
  else if ( *(_QWORD *)a2 != 10 )
  {
    goto LABEL_4;
  }
  if ( *(_QWORD *)(a2 + 24) == 0x64656D616E65525FLL && *(_WORD *)(a2 + 32) == 11822 )
  {
LABEL_12:
    v52 = "invalid symbol name from source";
    LOWORD(v55[1]) = 259;
    sub_E66880(a1, 0, (__int64)&v52);
  }
LABEL_4:
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, char *, size_t))(**(_QWORD **)(a1 + 152) + 48LL))(
          *(_QWORD *)(a1 + 152),
          v5,
          v4) )
  {
    n = 0;
    src = v51;
    v50 = 128;
    if ( v4 > 0x80 )
    {
      sub_C8D290((__int64)&src, v51, v4, 1u, v6, v7);
      v21 = (char *)src + n;
    }
    else
    {
      if ( !v4 )
      {
LABEL_15:
        v12 = (const char *)v55;
        v13 = 0;
        v52 = (const char *)v55;
        v54 = 128;
        qmemcpy(v55, "_Renamed..", 10);
        v14 = 10;
LABEL_16:
        v14 += v13;
LABEL_17:
        v53 = v14;
        v15 = sub_E6B3F0(a1, v12, v14);
        *(_BYTE *)(v15 + 20) = 1;
        v16 = (char *)v15;
        v17 = v15;
        v18 = sub_EA1530(80, v15, a1);
        v9 = v18;
        if ( v18 )
        {
          *(_QWORD *)(v18 + 24) = 0;
          *(_QWORD *)v18 = 0;
          v19 = *(_QWORD *)(v18 + 8) & 0xFFFF0000FFF00000LL;
          *(_QWORD *)(v18 - 8) = v17;
          *(_BYTE *)(v18 + 33) = 0;
          *(_BYTE *)(v18 + 35) = 0;
          *(_BYTE *)(v18 + 72) = 0;
          *(_QWORD *)(v18 + 8) = v19 | (2LL * a3) | 0x601;
          *(_DWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 40) = 0;
          *(_WORD *)(v18 + 48) = 0;
          *(_QWORD *)(v18 + 56) = 0;
          *(_QWORD *)(v18 + 64) = 0;
        }
        v20 = v5[v4 - 1] == 93;
        v41 = v5;
        v42 = v4;
        if ( v20 )
        {
          v16 = &v40;
          v40 = 91;
          v31 = sub_C93460((__int64 *)&v41, &v40, 1u);
          v5 = v41;
          v4 = v31;
          if ( v31 == -1 )
          {
            v4 = v42;
          }
          else if ( v42 <= v31 )
          {
            v4 = v42;
          }
        }
        *(_QWORD *)(v9 + 56) = v5;
        *(_QWORD *)(v9 + 64) = v4;
        *(_BYTE *)(v9 + 72) = 1;
        if ( v52 != (const char *)v55 )
          _libc_free(v52, v16);
        if ( src != v51 )
          _libc_free(src, v16);
        return v9;
      }
      v21 = v51;
    }
    memcpy(v21, v5, v4);
    v20 = v4 + n == 0;
    n += v4;
    v22 = (char *)src;
    if ( v20 )
      goto LABEL_15;
    v23 = "_Renamed..";
    v24 = (*(_BYTE *)src == 46) + 10LL;
    if ( *(_BYTE *)src == 46 )
      v23 = "._Renamed..";
    v34 = *(_BYTE *)src == 46;
    v52 = (const char *)v55;
    v54 = 128;
    *(_QWORD *)((char *)&v55[-1] + v24) = *(_QWORD *)&v23[v24 - 8];
    v25 = 0;
    do
    {
      v26 = v25;
      v25 += 8;
      *(_QWORD *)((char *)v55 + v26) = *(_QWORD *)&v23[v26];
    }
    while ( v25 < (((_DWORD)v24 - 1) & 0xFFFFFFF8) );
    v53 = v24;
    v27 = 0;
    do
    {
      while ( (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 152) + 40LL))(
                *(_QWORD *)(a1 + 152),
                (unsigned int)v22[v27]) )
      {
        v22 = (char *)src;
        if ( *((_BYTE *)src + v27) == 95 )
          break;
        v13 = n;
        if ( ++v27 >= n )
          goto LABEL_38;
      }
      v46 = 0x100000000LL;
      v42 = 2;
      v43 = 0;
      v41 = (char *)&unk_49DD288;
      v47 = &v52;
      v44 = 0;
      v45 = 0;
      sub_CB5980((__int64)&v41, 0, 0, 0);
      sub_CB5A50((__int64)&v41, *((char *)src + v27));
      v41 = (char *)&unk_49DD388;
      sub_CB5840((__int64)&v41);
      *((_BYTE *)src + v27) = 95;
      v13 = n;
      ++v27;
      v22 = (char *)src;
    }
    while ( v27 < n );
LABEL_38:
    v14 = v53;
    if ( !v34 )
    {
      if ( v54 < v13 + v53 )
      {
        v32 = v22;
        v36 = v13;
        sub_C8D290((__int64)&v52, v55, v13 + v53, 1u, v13 + v53, (__int64)v22);
        v14 = v53;
        v22 = v32;
        v13 = v36;
      }
      v12 = v52;
      if ( v13 )
      {
        v37 = v13;
        memcpy((void *)&v52[v14], v22, v13);
        v14 = v53;
        v12 = v52;
        v13 = v37;
      }
      goto LABEL_16;
    }
    if ( v13 )
    {
      --v13;
      ++v22;
      v28 = v13 + v53;
      if ( v54 >= v13 + v53 )
      {
LABEL_41:
        v12 = v52;
        if ( !v13 )
          goto LABEL_17;
        v35 = v13;
        memcpy((void *)&v52[v14], v22, v13);
        v14 = v35 + v53;
LABEL_43:
        v12 = v52;
        goto LABEL_17;
      }
    }
    else
    {
      if ( v54 >= v53 )
        goto LABEL_43;
      v28 = v53;
    }
    v33 = v22;
    v38 = v13;
    sub_C8D290((__int64)&v52, v55, v28, 1u, v28, (__int64)v22);
    v14 = v53;
    v22 = v33;
    v13 = v38;
    goto LABEL_41;
  }
  v8 = sub_EA1530(80, a2, a1);
  v9 = v8;
  if ( v8 )
  {
    *(_QWORD *)(v8 + 24) = 0;
    *(_QWORD *)v8 = 0;
    v10 = *(_QWORD *)(v8 + 8) & 0xFFFF0000FFF00000LL;
    *(_QWORD *)(v8 - 8) = a2;
    *(_DWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 8) = v10 | (2LL * a3) | 0x601;
LABEL_7:
    *(_BYTE *)(v9 + 33) = 0;
    *(_BYTE *)(v9 + 35) = 0;
    *(_QWORD *)(v9 + 40) = 0;
    *(_WORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = 0;
    *(_QWORD *)(v9 + 64) = 0;
    *(_BYTE *)(v9 + 72) = 0;
  }
  return v9;
}
