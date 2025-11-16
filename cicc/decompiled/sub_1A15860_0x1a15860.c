// Function: sub_1A15860
// Address: 0x1a15860
//
void __fastcall sub_1A15860(__int64 a1, __int64 a2)
{
  int v4; // r8d
  int v5; // r9d
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int8 v8; // al
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  _BYTE *v12; // rdi
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned int v15; // esi
  __int64 v16; // rax
  char *v17; // rax
  char *i; // rdx
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  unsigned int v24; // r14d
  bool v25; // al
  unsigned int v26; // eax
  int v27; // r8d
  int v28; // r9d
  size_t v29; // r15
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  int v35; // r8d
  __int64 v36; // rdi
  __int64 j; // rax
  unsigned int v38; // ecx
  __int64 v39; // rsi
  unsigned __int64 v40; // rsi
  __int64 v41; // r15
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // r14
  unsigned int v45; // edx
  __int64 v46; // rdi
  __int64 v47; // r10
  __int64 v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rcx
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rdx
  char v54; // dl
  __int64 v55; // rdi
  __int64 v56; // rdx
  unsigned int v57; // eax
  int v58; // r8d
  int v59; // r9d
  size_t v60; // r13
  unsigned int v61; // r15d
  __int64 v62; // rcx
  __int64 v63; // r9
  __int64 v64; // rdi
  __int64 v65; // r9
  __int64 v66; // rdi
  void *s; // [rsp+10h] [rbp-50h] BYREF
  __int64 v68; // [rsp+18h] [rbp-48h]
  _BYTE v69[64]; // [rsp+20h] [rbp-40h] BYREF

  s = v69;
  v68 = 0x1000000000LL;
  v6 = (unsigned int)sub_15F4D60(a2);
  v7 = 0;
  if ( v6 )
  {
    if ( v6 > HIDWORD(v68) )
    {
      sub_16CD150((__int64)&s, v69, v6, 1, v4, v5);
      v7 = (unsigned int)v68;
    }
    v17 = (char *)s + v7;
    for ( i = (char *)s + v6; i != v17; ++v17 )
    {
      if ( v17 )
        *v17 = 0;
    }
    v8 = *(_BYTE *)(a2 + 16);
    LODWORD(v68) = v6;
    if ( v8 == 26 )
    {
LABEL_3:
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
      {
        v9 = *sub_1A10F60(a1, *(_QWORD *)(a2 - 72));
        v10 = (v9 >> 1) & 3;
        if ( v10 == 1 || v10 == 2 )
        {
          v23 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_BYTE *)(v23 + 16) == 13 )
          {
            v24 = *(_DWORD *)(v23 + 32);
            if ( v24 <= 0x40 )
              v25 = *(_QWORD *)(v23 + 24) == 0;
            else
              v25 = v24 == (unsigned int)sub_16A57B0(v23 + 24);
            *((_BYTE *)s + v25) = 1;
            goto LABEL_9;
          }
        }
        if ( !v10 )
        {
LABEL_9:
          v11 = (unsigned int)v68;
          v12 = s;
          goto LABEL_10;
        }
        *((_BYTE *)s + 1) = 1;
      }
      *(_BYTE *)s = 1;
      goto LABEL_9;
    }
  }
  else
  {
    v8 = *(_BYTE *)(a2 + 16);
    if ( v8 == 26 )
      goto LABEL_3;
  }
  v19 = v8 - 24;
  if ( v19 > 6 )
  {
    if ( (unsigned int)v8 - 32 <= 2 )
      goto LABEL_40;
  }
  else if ( v19 > 4 )
  {
    goto LABEL_40;
  }
  if ( v8 != 27 )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v30 = *(__int64 **)(a2 - 8);
    else
      v30 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v31 = *sub_1A10F60(a1, *v30);
    v32 = (v31 >> 1) & 3;
    if ( (v32 == 1 || v32 == 2) && (v33 = v31 & 0xFFFFFFFFFFFFFFF8LL, *(_BYTE *)(v33 + 16) == 4) )
    {
      v34 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v35 = v34 - 1;
      v36 = a2 - 24 * v34;
      for ( j = 0; v35 != (_DWORD)j; j = v38 )
      {
        v38 = j + 1;
        v39 = v36;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v39 = *(_QWORD *)(a2 - 8);
        if ( *(_QWORD *)(v33 - 24) == *(_QWORD *)(v39 + 24LL * v38) )
        {
          *((_BYTE *)s + j) = 1;
          break;
        }
      }
    }
    else if ( v32 )
    {
      v57 = sub_15F4D60(a2);
      LODWORD(v68) = 0;
      v60 = v57;
      v61 = v57;
      if ( HIDWORD(v68) < v57 )
        sub_16CD150((__int64)&s, v69, v57, 1, v58, v59);
      LODWORD(v68) = v61;
      v12 = s;
      if ( v60 )
      {
        memset(s, 1, v60);
        v12 = s;
      }
      goto LABEL_50;
    }
    v12 = s;
LABEL_50:
    v11 = (unsigned int)v68;
    goto LABEL_10;
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 == 1 )
  {
    *(_BYTE *)s = 1;
    v11 = (unsigned int)v68;
    v12 = s;
    goto LABEL_10;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v20 = *(__int64 **)(a2 - 8);
  else
    v20 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v21 = *sub_1A10F60(a1, *v20);
  v22 = (v21 >> 1) & 3;
  if ( v22 == 2 || v22 == 1 )
  {
    v40 = v21 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)(v40 + 16) == 13 )
    {
      v41 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v42 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
      v43 = v42 >> 2;
      if ( v42 >> 2 )
      {
        v44 = 4 * v43;
        v43 = 0;
        v45 = 8;
        while ( 1 )
        {
          v47 = v43 + 1;
          v50 = a2 - 24LL * (unsigned int)v41;
          if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
            v50 = *(_QWORD *)(a2 - 8);
          v51 = *(_QWORD *)(v50 + 24LL * (v45 - 6));
          if ( v51 )
          {
            if ( v40 == v51 )
              break;
          }
          v46 = *(_QWORD *)(v50 + 24LL * (v45 - 4));
          if ( v46 && v40 == v46 )
          {
LABEL_81:
            v43 = v47;
            break;
          }
          v47 = v43 + 3;
          v48 = *(_QWORD *)(v50 + 24LL * (v45 - 2));
          if ( v48 && v40 == v48 )
          {
            v43 += 2;
            break;
          }
          v43 += 4;
          v49 = *(_QWORD *)(v50 + 24LL * v45);
          if ( v49 && v40 == v49 )
            goto LABEL_81;
          v45 += 8;
          if ( v43 == v44 )
          {
            v53 = v42 - v43;
            goto LABEL_83;
          }
        }
LABEL_76:
        if ( v43 != v42 && (_DWORD)v43 != -2 )
        {
          v52 = (unsigned int)(v43 + 1);
LABEL_79:
          *((_BYTE *)s + v52) = 1;
          v11 = (unsigned int)v68;
          v12 = s;
          goto LABEL_10;
        }
LABEL_91:
        v52 = 0;
        goto LABEL_79;
      }
      v53 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
LABEL_83:
      switch ( v53 )
      {
        case 2LL:
          v62 = v43;
          v54 = *(_BYTE *)(a2 + 23) & 0x40;
          break;
        case 3LL:
          v62 = v43 + 1;
          v54 = *(_BYTE *)(a2 + 23) & 0x40;
          if ( v54 )
            v63 = *(_QWORD *)(a2 - 8);
          else
            v63 = a2 - 24LL * (unsigned int)v41;
          v64 = *(_QWORD *)(v63 + 24LL * (unsigned int)(2 * (v43 + 1)));
          if ( v64 && v40 == v64 )
            goto LABEL_76;
          break;
        case 1LL:
          v54 = *(_BYTE *)(a2 + 23) & 0x40;
          goto LABEL_87;
        default:
          goto LABEL_91;
      }
      v43 = v62 + 1;
      if ( v54 )
        v65 = *(_QWORD *)(a2 - 8);
      else
        v65 = a2 - 24LL * (unsigned int)v41;
      v66 = *(_QWORD *)(v65 + 24LL * (unsigned int)(2 * (v62 + 1)));
      if ( v66 && v40 == v66 )
      {
        v43 = v62;
        goto LABEL_76;
      }
LABEL_87:
      if ( v54 )
        v55 = *(_QWORD *)(a2 - 8);
      else
        v55 = a2 - 24 * v41;
      v56 = *(_QWORD *)(v55 + 24LL * (unsigned int)(2 * v43 + 2));
      if ( !v56 || v40 != v56 )
        goto LABEL_91;
      goto LABEL_76;
    }
  }
  if ( !v22 )
    goto LABEL_9;
LABEL_40:
  v26 = sub_15F4D60(a2);
  LODWORD(v68) = 0;
  v29 = v26;
  v11 = v26;
  if ( HIDWORD(v68) < v26 )
    sub_16CD150((__int64)&s, v69, v26, 1, v27, v28);
  LODWORD(v68) = v11;
  v12 = (char *)s + v29;
  if ( s != (char *)s + v29 )
  {
    memset(s, 1, v29);
    goto LABEL_9;
  }
LABEL_10:
  v13 = *(_QWORD *)(a2 + 40);
  if ( (_DWORD)v11 )
  {
    v14 = 0;
    do
    {
      while ( !v12[v14] )
      {
        if ( ++v14 == v11 )
          goto LABEL_15;
      }
      v15 = v14++;
      v16 = sub_15F4DF0(a2, v15);
      sub_1A153A0(a1, v13, v16);
      v12 = s;
    }
    while ( v14 != v11 );
  }
LABEL_15:
  if ( v12 != v69 )
    _libc_free((unsigned __int64)v12);
}
