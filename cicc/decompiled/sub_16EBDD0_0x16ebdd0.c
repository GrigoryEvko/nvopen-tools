// Function: sub_16EBDD0
// Address: 0x16ebdd0
//
__int64 __fastcall sub_16EBDD0(__int64 a1, const char *a2, int a3)
{
  int v3; // eax
  char v4; // r13
  size_t v5; // r14
  unsigned __int64 v6; // r12
  __int64 v7; // r15
  void *v8; // rax
  __int64 v9; // r8
  signed __int64 v10; // r9
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  signed __int64 v13; // rsi
  signed __int64 v14; // rdx
  __int64 v15; // rcx
  const char *v16; // rax
  signed __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // r10
  int v20; // edx
  int v21; // ecx
  _BYTE *v22; // rax
  int v23; // edx
  int v24; // eax
  signed __int64 v25; // r13
  char v26; // r14
  int v27; // edx
  int v28; // edi
  int v29; // edi
  _BYTE *v30; // rax
  int v31; // edx
  unsigned int v32; // eax
  int v33; // r13d
  __int64 v34; // r14
  __int64 result; // rax
  unsigned __int64 v36; // r14
  char *v37; // rax
  __int64 *v38; // r10
  __int64 *v39; // r15
  __int64 *v40; // rbx
  __int64 *i; // rcx
  __int64 v42; // rax
  __int64 *v43; // r11
  unsigned __int64 v44; // rdx
  int v45; // edx
  _BYTE *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  int v50; // [rsp+8h] [rbp-128h]
  int v51; // [rsp+10h] [rbp-120h]
  signed __int64 v52; // [rsp+10h] [rbp-120h]
  const char *v54; // [rsp+20h] [rbp-110h] BYREF
  const char *v55; // [rsp+28h] [rbp-108h]
  unsigned int v56; // [rsp+30h] [rbp-100h]
  void *src; // [rsp+38h] [rbp-F8h]
  signed __int64 v58; // [rsp+40h] [rbp-F0h]
  signed __int64 v59; // [rsp+48h] [rbp-E8h]
  int v60; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v61; // [rsp+58h] [rbp-D8h]
  _BYTE v62[160]; // [rsp+60h] [rbp-D0h] BYREF

  v3 = a3;
  LOBYTE(v3) = a3 & 0x7F;
  v51 = v3;
  if ( (a3 & 0x11) == 0x11 )
    return 16;
  v4 = a3;
  if ( (a3 & 0x20) != 0 )
  {
    v36 = *(_QWORD *)(a1 + 16);
    result = 16;
    if ( v36 < (unsigned __int64)a2 )
      return result;
    v5 = v36 - (_QWORD)a2;
  }
  else
  {
    v5 = strlen(a2);
  }
  v6 = malloc(0x18Fu);
  if ( !v6 )
    return 12;
  v7 = (v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL) + 1;
  v58 = v7;
  v8 = _libc_calloc(v7, 8u);
  v10 = (v5 >> 1) + (v5 & 0xFFFFFFFFFFFFFFFELL);
  v59 = 0;
  src = v8;
  v11 = v8;
  if ( !v8 )
  {
    _libc_free(v6);
    return 12;
  }
  v54 = a2;
  v55 = &a2[v5];
  memset(v62, 0, sizeof(v62));
  v61 = v6;
  *(_DWORD *)(v6 + 40) = v51;
  *(_QWORD *)(v6 + 88) = v6 + 264;
  *(_QWORD *)(v6 + 16) = 256;
  *(_QWORD *)(v6 + 24) = 0;
  *(_QWORD *)(v6 + 32) = 0;
  *(_QWORD *)(v6 + 96) = 0;
  *(_DWORD *)(v6 + 104) = 0;
  *(_QWORD *)(v6 + 112) = 0;
  *(_QWORD *)(v6 + 72) = 0;
  *(_QWORD *)(v6 + 80) = 0x100000000LL;
  v56 = 0;
  memset((void *)(v6 + 136), 0, 0x100u);
  v12 = 1;
  v60 = 0;
  *(_DWORD *)(v6 + 120) = 0;
  if ( v7 <= 0 )
  {
    v10 += 2LL;
    v13 = ((v10 + ((unsigned __int64)v10 >> 63)) & 0xFFFFFFFFFFFFFFFELL) + v10 / 2;
    if ( v7 < v13 )
    {
      sub_16E90A0((__int64)&v54, v13, (int)v8, 1, v9, v10);
      v12 = v59 + 1;
      v11 = (char *)src + 8 * v59;
    }
  }
  v59 = v12;
  *v11 = 0x8000000;
  v14 = v59;
  v15 = v59 - 1;
  *(_QWORD *)(v6 + 56) = v59 - 1;
  if ( (v4 & 1) != 0 )
  {
    v17 = 128;
    sub_16EAF00((__int64)&v54, 128, v14, v15, v9, v10);
    v18 = v56;
    v14 = v59;
  }
  else
  {
    if ( (v4 & 0x10) != 0 )
    {
      v16 = v54;
      if ( v54 >= v55 )
      {
        v17 = v56;
        if ( !v56 )
          v56 = 14;
        v54 = (const char *)&unk_4FA17D0;
        v55 = (const char *)&unk_4FA17D0;
        goto LABEL_15;
      }
      do
      {
        v54 = v16 + 1;
        v17 = (unsigned int)*v16;
        sub_16EA3B0((__int64)&v54, v17, (__int64)(v16 + 1), v15, v9, v10);
        v16 = v54;
      }
      while ( v54 < v55 );
    }
    else
    {
      v17 = 128;
      sub_16EA500((__int64)&v54, 128, 128, v15, v9, v10);
    }
    v18 = v56;
    v14 = v59;
  }
  v15 = v14 - 1;
  if ( !v18 )
  {
    if ( v58 <= v14 )
    {
      v17 = ((v58 + 1 + ((unsigned __int64)(v58 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v58 + 1) / 2;
      if ( v58 < v17 )
      {
        sub_16E90A0((__int64)&v54, v17, v14, v58, v9, v10);
        v14 = v59;
      }
    }
    v59 = v14 + 1;
    *((_QWORD *)src + v14) = 0x8000000;
    v14 = v59;
    v15 = v56;
    *(_QWORD *)(v6 + 64) = v59 - 1;
    if ( (_DWORD)v15 )
      goto LABEL_42;
    v19 = *(_QWORD *)(v6 + 88);
    v10 = -127;
    v17 = 4294967168LL;
    if ( !*(_BYTE *)(v19 - 128) )
      goto LABEL_24;
    while ( 1 )
    {
      do
      {
LABEL_40:
        if ( (_DWORD)v10 == 128 )
        {
          v14 = v59;
          goto LABEL_42;
        }
LABEL_39:
        v17 = (unsigned int)v10++;
      }
      while ( *(_BYTE *)(v19 + v10 - 1) );
LABEL_24:
      v20 = *(_DWORD *)(v6 + 20);
      v17 = (unsigned __int8)v17;
      v21 = v20 + 14;
      if ( v20 + 7 >= 0 )
        v21 = v20 + 7;
      v15 = (unsigned int)(v21 >> 3);
      if ( v20 > 0 )
      {
        v22 = (_BYTE *)((unsigned __int8)v17 + *(_QWORD *)(v6 + 32));
        v23 = 0;
        while ( !*v22 )
        {
          ++v23;
          v22 += *(int *)(v6 + 16);
          if ( (int)v15 <= v23 )
            goto LABEL_40;
        }
        v24 = *(_DWORD *)(v6 + 84);
        v25 = v10;
        v26 = v24;
        *(_DWORD *)(v6 + 84) = v24 + 1;
        *(_BYTE *)(v19 + v10 - 1) = v24;
        if ( (int)v10 <= 127 )
          break;
      }
    }
    while ( 1 )
    {
      while ( *(_BYTE *)(v19 + v25) )
      {
LABEL_38:
        if ( (int)++v25 > 127 )
          goto LABEL_39;
      }
      v27 = *(_DWORD *)(v6 + 20);
      v28 = v27 + 14;
      v15 = (unsigned int)(v27 + 7);
      if ( v27 + 7 >= 0 )
        v28 = v27 + 7;
      v29 = v28 >> 3;
      if ( v27 > 0 )
      {
        v30 = (_BYTE *)((unsigned __int8)v17 + *(_QWORD *)(v6 + 32));
        v31 = 0;
        do
        {
          v15 = (unsigned __int8)v30[(unsigned __int8)v25 - (unsigned __int8)v17];
          if ( *v30 != (_BYTE)v15 )
            goto LABEL_38;
          ++v31;
          v30 += *(int *)(v6 + 16);
        }
        while ( v29 > v31 );
      }
      *(_BYTE *)(v19 + v25++) = v26;
      if ( (int)v25 > 127 )
        goto LABEL_39;
    }
  }
LABEL_15:
  *(_QWORD *)(v6 + 64) = v15;
LABEL_42:
  *(_QWORD *)(v6 + 48) = v14;
  if ( (unsigned __int64)v14 > 0x1FFFFFFFFFFFFFFFLL )
  {
    v32 = v56;
    *(_QWORD *)(v6 + 8) = src;
    if ( !v32 )
      v56 = 12;
    v33 = *(_DWORD *)(v6 + 72);
    v34 = *(_QWORD *)(v6 + 112);
    v54 = (const char *)&unk_4FA17D0;
    v55 = (const char *)&unk_4FA17D0;
    goto LABEL_46;
  }
  v17 = 8 * v14;
  v37 = realloc((unsigned __int64)src, 8 * v14, v14, v15, v9, v10);
  *(_QWORD *)(v6 + 8) = v37;
  if ( !v37 )
  {
    if ( !v56 )
      v56 = 12;
    v54 = (const char *)&unk_4FA17D0;
    v55 = (const char *)&unk_4FA17D0;
    *(_QWORD *)(v6 + 8) = src;
  }
  v33 = *(_DWORD *)(v6 + 72);
  v34 = *(_QWORD *)(v6 + 112);
  if ( v56 )
  {
LABEL_46:
    *(_QWORD *)(v6 + 128) = 0;
    *(_DWORD *)v6 = 53829;
    *(_QWORD *)(a1 + 8) = v34;
    *(_QWORD *)(a1 + 24) = v6;
    *(_DWORD *)a1 = 62053;
    if ( (v33 & 4) != 0 )
      goto LABEL_87;
    goto LABEL_47;
  }
  v10 = 0;
  v38 = 0;
  v39 = 0;
  v17 = 2415919104LL;
  v40 = (__int64 *)(*(_QWORD *)(v6 + 8) + 8LL);
  for ( i = v40; ; i = v43 )
  {
    v42 = *i;
    v43 = i + 1;
    v44 = *i & 0xF8000000LL;
    if ( v44 == 1476395008 )
    {
LABEL_76:
      while ( 1 )
      {
        i += v42 & 0x7FFFFFF;
        v42 = *i;
        v44 = *i & 0xF8000000LL;
        if ( v44 == 1610612736 || v44 == 2415919104 )
          goto LABEL_89;
        if ( v44 != 2281701376 )
        {
          v33 |= 4u;
          *(_DWORD *)(v6 + 72) = v33;
          goto LABEL_80;
        }
      }
    }
    if ( v44 <= 0x58000000 )
      break;
    if ( v44 != 1879048192 )
    {
      if ( v44 == 2013265920 )
        goto LABEL_76;
      if ( v44 != 1744830464 )
      {
        if ( v10 > *(int *)(v6 + 104) )
          goto LABEL_67;
        goto LABEL_105;
      }
    }
LABEL_60:
    ;
  }
  if ( v44 == 0x10000000 )
  {
    if ( !v10 )
      v38 = i;
    ++v10;
    goto LABEL_60;
  }
  ++i;
  if ( v44 == 1207959552 )
    goto LABEL_60;
LABEL_89:
  v43 = i;
  if ( v10 > *(int *)(v6 + 104) )
  {
LABEL_67:
    *(_DWORD *)(v6 + 104) = v10;
    v39 = v38;
  }
  if ( v44 != 0x8000000 )
  {
LABEL_105:
    v10 = 0;
    goto LABEL_60;
  }
  v45 = *(_DWORD *)(v6 + 104);
  if ( v45 )
  {
    v50 = *(_DWORD *)(v6 + 104);
    v52 = v45;
    v46 = (_BYTE *)malloc(v45 + 1LL);
    v17 = v52;
    *(_QWORD *)(v6 + 96) = v46;
    if ( v46 )
    {
      if ( v50 <= 0 )
      {
        v17 = (signed __int64)v46;
      }
      else
      {
        v17 = (signed __int64)&v46[v52];
        do
        {
          do
            v47 = *v39++;
          while ( (v47 & 0xF8000000) != 0x10000000 );
          *v46++ = v47;
        }
        while ( v46 != (_BYTE *)v17 );
      }
      *(_BYTE *)v17 = 0;
    }
    else
    {
      *(_DWORD *)(v6 + 104) = 0;
    }
  }
LABEL_80:
  v15 = 0;
  v14 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v48 = *v40++;
        v49 = (unsigned int)v48 & 0xF8000000;
        if ( v49 != 1207959552 )
          break;
        ++v14;
      }
      if ( v49 != 1342177280 )
        break;
      if ( v15 < v14 )
        v15 = v14;
      --v14;
    }
  }
  while ( v49 != 0x8000000 );
  if ( v14 )
  {
    *(_QWORD *)(v6 + 128) = v15;
    *(_DWORD *)(v6 + 72) = v33 | 4;
    *(_DWORD *)v6 = 53829;
    *(_QWORD *)(a1 + 8) = v34;
    *(_QWORD *)(a1 + 24) = v6;
    *(_DWORD *)a1 = 62053;
    goto LABEL_86;
  }
  *(_QWORD *)(v6 + 128) = v15;
  *(_DWORD *)v6 = 53829;
  *(_QWORD *)(a1 + 8) = v34;
  *(_QWORD *)(a1 + 24) = v6;
  *(_DWORD *)a1 = 62053;
  result = v33 & 4;
  if ( (v33 & 4) != 0 )
  {
LABEL_86:
    v56 = 15;
LABEL_87:
    v54 = (const char *)&unk_4FA17D0;
    v55 = (const char *)&unk_4FA17D0;
LABEL_47:
    sub_16F05C0(a1, v17, v14, v15, v9, v10);
    return v56;
  }
  return result;
}
