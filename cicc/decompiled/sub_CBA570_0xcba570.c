// Function: sub_CBA570
// Address: 0xcba570
//
__int64 __fastcall sub_CBA570(__int64 a1, const char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  char v7; // r13
  size_t v8; // r14
  __int64 v9; // r12
  signed __int64 v10; // r15
  _QWORD *v11; // rdx
  __int64 v12; // rcx
  signed __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  unsigned __int8 *v16; // rax
  unsigned int v17; // eax
  signed __int64 v18; // rsi
  unsigned int v19; // ecx
  __int64 v20; // r10
  __int64 v21; // r9
  unsigned __int8 v22; // si
  int v23; // edx
  int v24; // ecx
  int v25; // ecx
  _BYTE *v26; // rax
  int v27; // edx
  int v28; // eax
  __int64 v29; // r13
  char v30; // r14
  int v31; // edx
  int v32; // edi
  int v33; // edi
  _BYTE *v34; // rax
  int v35; // edx
  unsigned int v36; // eax
  int v37; // r13d
  __int64 v38; // r14
  __int64 result; // rax
  unsigned __int64 v40; // r14
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 *v44; // r10
  __int64 *v45; // r15
  __int64 *v46; // rbx
  __int64 *i; // rcx
  __int64 v48; // rax
  __int64 *v49; // r11
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  _BYTE *v52; // rax
  _BYTE *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  int v59; // [rsp+8h] [rbp-128h]
  int v60; // [rsp+10h] [rbp-120h]
  __int64 v61; // [rsp+10h] [rbp-120h]
  unsigned __int8 *v63; // [rsp+20h] [rbp-110h] BYREF
  unsigned __int8 *v64; // [rsp+28h] [rbp-108h]
  unsigned int v65; // [rsp+30h] [rbp-100h]
  void *src; // [rsp+38h] [rbp-F8h]
  signed __int64 v67; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v68; // [rsp+48h] [rbp-E8h]
  int v69; // [rsp+50h] [rbp-E0h]
  __int64 v70; // [rsp+58h] [rbp-D8h]
  _BYTE v71[160]; // [rsp+60h] [rbp-D0h] BYREF

  v6 = a3;
  LOBYTE(v6) = a3 & 0x7F;
  v60 = v6;
  if ( (a3 & 0x11) == 0x11 )
    return 16;
  v7 = a3;
  if ( (a3 & 0x20) != 0 )
  {
    v40 = *(_QWORD *)(a1 + 16);
    result = 16;
    if ( v40 < (unsigned __int64)a2 )
      return result;
    v8 = v40 - (_QWORD)a2;
  }
  else
  {
    v8 = strlen(a2);
  }
  v9 = malloc(399, a2, a3, a4, a5, a6);
  if ( !v9 )
    return 12;
  v10 = (v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL) + 1;
  v67 = v10;
  v68 = 0;
  src = (void *)_libc_calloc(v10, 8);
  v11 = src;
  if ( !src )
  {
    _libc_free(v9, 8);
    return 12;
  }
  v63 = (unsigned __int8 *)a2;
  v64 = (unsigned __int8 *)&a2[v8];
  memset(v71, 0, sizeof(v71));
  v70 = v9;
  *(_DWORD *)(v9 + 40) = v60;
  *(_QWORD *)(v9 + 88) = v9 + 264;
  *(_QWORD *)(v9 + 16) = 256;
  *(_QWORD *)(v9 + 24) = 0;
  *(_QWORD *)(v9 + 32) = 0;
  *(_QWORD *)(v9 + 96) = 0;
  *(_DWORD *)(v9 + 104) = 0;
  *(_QWORD *)(v9 + 112) = 0;
  *(_QWORD *)(v9 + 72) = 0;
  *(_QWORD *)(v9 + 80) = 0x100000000LL;
  v65 = 0;
  memset((void *)(v9 + 136), 0, 0x100u);
  v12 = 1;
  v69 = 0;
  *(_DWORD *)(v9 + 120) = 0;
  if ( v10 <= 0 )
  {
    v13 = (((v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL) + 2 + (((v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL) + 2) >> 63))
         & 0xFFFFFFFFFFFFFFFELL)
        + (__int64)((v8 >> 1) + (v8 & 0xFFFFFFFFFFFFFFFELL) + 2) / 2;
    if ( v10 < v13 )
    {
      sub_CB7740((__int64)&v63, v13);
      v12 = v68 + 1;
      v11 = (char *)src + 8 * v68;
    }
  }
  v68 = v12;
  *v11 = 0x8000000;
  v14 = v68;
  v15 = v68 - 1;
  *(_QWORD *)(v9 + 56) = v68 - 1;
  if ( (v7 & 1) != 0 )
  {
    sub_CB9640((__int64)&v63, 128);
    v17 = v65;
    v14 = v68;
  }
  else
  {
    if ( (v7 & 0x10) != 0 )
    {
      v16 = v63;
      if ( v64 - v63 <= 0 )
      {
        if ( !v65 )
          v65 = 14;
        v63 = byte_4F85140;
        v64 = byte_4F85140;
        goto LABEL_96;
      }
      do
      {
        v63 = v16 + 1;
        sub_CB8AB0((__int64)&v63, (unsigned int)(char)*v16);
        v16 = v63;
      }
      while ( v64 - v63 > 0 );
    }
    else
    {
      sub_CB8C00((__int64)&v63, 128, 128);
    }
    v17 = v65;
    v14 = v68;
  }
  v15 = v14 - 1;
  if ( v17 )
  {
LABEL_96:
    *(_QWORD *)(v9 + 64) = v15;
    goto LABEL_38;
  }
  if ( (__int64)v14 >= v67 )
  {
    v18 = ((v67 + 1 + ((unsigned __int64)(v67 + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL) + (v67 + 1) / 2;
    if ( v67 < v18 )
    {
      sub_CB7740((__int64)&v63, v18);
      v14 = v68;
    }
  }
  v68 = v14 + 1;
  *((_QWORD *)src + v14) = 0x8000000;
  v14 = v68;
  v19 = v65;
  *(_QWORD *)(v9 + 64) = v68 - 1;
  if ( v19 )
    goto LABEL_38;
  v20 = *(_QWORD *)(v9 + 88);
  v21 = -127;
  v22 = 0x80;
  if ( !*(_BYTE *)(v20 - 128) )
    goto LABEL_20;
LABEL_36:
  while ( (_DWORD)v21 != 128 )
  {
LABEL_35:
    v22 = v21++;
    if ( !*(_BYTE *)(v20 + v21 - 1) )
    {
LABEL_20:
      v23 = *(_DWORD *)(v9 + 20);
      v24 = v23 + 14;
      if ( v23 + 7 >= 0 )
        v24 = v23 + 7;
      v25 = v24 >> 3;
      if ( v23 > 0 )
      {
        v26 = (_BYTE *)(v22 + *(_QWORD *)(v9 + 32));
        v27 = 0;
        while ( !*v26 )
        {
          ++v27;
          v26 += *(int *)(v9 + 16);
          if ( v25 <= v27 )
            goto LABEL_36;
        }
        v28 = *(_DWORD *)(v9 + 84);
        v29 = v21;
        v30 = v28;
        *(_DWORD *)(v9 + 84) = v28 + 1;
        *(_BYTE *)(v20 + v21 - 1) = v28;
        if ( (int)v21 <= 127 )
        {
          while ( 1 )
          {
            while ( *(_BYTE *)(v20 + v29) )
            {
LABEL_34:
              if ( (int)++v29 > 127 )
                goto LABEL_35;
            }
            v31 = *(_DWORD *)(v9 + 20);
            v32 = v31 + 14;
            if ( v31 + 7 >= 0 )
              v32 = v31 + 7;
            v33 = v32 >> 3;
            if ( v31 > 0 )
            {
              v34 = (_BYTE *)(v22 + *(_QWORD *)(v9 + 32));
              v35 = 0;
              while ( *v34 == v34[(unsigned __int8)v29 - v22] )
              {
                ++v35;
                v34 += *(int *)(v9 + 16);
                if ( v33 <= v35 )
                  goto LABEL_47;
              }
              goto LABEL_34;
            }
LABEL_47:
            *(_BYTE *)(v20 + v29++) = v30;
            if ( (int)v29 > 127 )
              goto LABEL_35;
          }
        }
      }
    }
  }
  v14 = v68;
LABEL_38:
  *(_QWORD *)(v9 + 48) = v14;
  if ( v14 > 0x1FFFFFFFFFFFFFFFLL )
  {
    v36 = v65;
    *(_QWORD *)(v9 + 8) = src;
    if ( !v36 )
      v65 = 12;
    v37 = *(_DWORD *)(v9 + 72);
    v38 = *(_QWORD *)(v9 + 112);
    v63 = byte_4F85140;
    v64 = byte_4F85140;
    goto LABEL_42;
  }
  v41 = realloc(src);
  *(_QWORD *)(v9 + 8) = v41;
  if ( !v41 )
  {
    if ( !v65 )
      v65 = 12;
    v63 = byte_4F85140;
    v64 = byte_4F85140;
    *(_QWORD *)(v9 + 8) = src;
  }
  v37 = *(_DWORD *)(v9 + 72);
  v38 = *(_QWORD *)(v9 + 112);
  if ( v65 )
  {
LABEL_42:
    *(_QWORD *)(v9 + 128) = 0;
    *(_DWORD *)v9 = 53829;
    *(_QWORD *)(a1 + 8) = v38;
    *(_QWORD *)(a1 + 24) = v9;
    *(_DWORD *)a1 = 62053;
    if ( (v37 & 4) != 0 )
      goto LABEL_83;
    goto LABEL_43;
  }
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = (__int64 *)(*(_QWORD *)(v9 + 8) + 8LL);
  for ( i = v46; ; i = v49 )
  {
    v48 = *i;
    v49 = i + 1;
    v50 = *i & 0xF8000000LL;
    if ( v50 == 1476395008 )
    {
LABEL_72:
      while ( 1 )
      {
        i += v48 & 0x7FFFFFF;
        v48 = *i;
        v50 = *i & 0xF8000000LL;
        if ( v50 == 1610612736 || v50 == 2415919104 )
          goto LABEL_85;
        if ( v50 != 2281701376 )
        {
          v37 |= 4u;
          *(_DWORD *)(v9 + 72) = v37;
          goto LABEL_76;
        }
      }
    }
    if ( v50 <= 0x58000000 )
      break;
    if ( v50 != 1879048192 )
    {
      if ( v50 == 2013265920 )
        goto LABEL_72;
      if ( v50 != 1744830464 )
      {
        if ( v43 > *(int *)(v9 + 104) )
          goto LABEL_63;
        goto LABEL_105;
      }
    }
LABEL_56:
    ;
  }
  if ( v50 == 0x10000000 )
  {
    if ( !v43 )
      v44 = i;
    ++v43;
    goto LABEL_56;
  }
  ++i;
  if ( v50 == 1207959552 )
    goto LABEL_56;
LABEL_85:
  v49 = i;
  if ( v43 > *(int *)(v9 + 104) )
  {
LABEL_63:
    *(_DWORD *)(v9 + 104) = v43;
    v45 = v44;
  }
  if ( v50 != 0x8000000 )
  {
LABEL_105:
    v43 = 0;
    goto LABEL_56;
  }
  v51 = *(unsigned int *)(v9 + 104);
  if ( (_DWORD)v51 )
  {
    v59 = *(_DWORD *)(v9 + 104);
    v61 = (int)v51;
    v52 = (_BYTE *)malloc((int)v51 + 1LL, (int)v51, v51, i, v42, v43);
    *(_QWORD *)(v9 + 96) = v52;
    if ( v52 )
    {
      if ( v59 <= 0 )
      {
        v53 = v52;
      }
      else
      {
        v53 = &v52[v61];
        do
        {
          do
            v54 = *v45++;
          while ( (v54 & 0xF8000000) != 0x10000000 );
          *v52++ = v54;
        }
        while ( v52 != v53 );
      }
      *v53 = 0;
    }
    else
    {
      *(_DWORD *)(v9 + 104) = 0;
    }
  }
LABEL_76:
  v55 = 0;
  v56 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v57 = *v46++;
        v58 = (unsigned int)v57 & 0xF8000000;
        if ( v58 != 1207959552 )
          break;
        ++v56;
      }
      if ( v58 != 1342177280 )
        break;
      if ( v55 < v56 )
        v55 = v56;
      --v56;
    }
  }
  while ( v58 != 0x8000000 );
  if ( v56 )
  {
    *(_QWORD *)(v9 + 128) = v55;
    *(_DWORD *)(v9 + 72) = v37 | 4;
    *(_DWORD *)v9 = 53829;
    *(_QWORD *)(a1 + 8) = v38;
    *(_QWORD *)(a1 + 24) = v9;
    *(_DWORD *)a1 = 62053;
    goto LABEL_82;
  }
  *(_QWORD *)(v9 + 128) = v55;
  *(_DWORD *)v9 = 53829;
  *(_QWORD *)(a1 + 8) = v38;
  *(_QWORD *)(a1 + 24) = v9;
  *(_DWORD *)a1 = 62053;
  result = v37 & 4;
  if ( (v37 & 4) != 0 )
  {
LABEL_82:
    v65 = 15;
LABEL_83:
    v63 = byte_4F85140;
    v64 = byte_4F85140;
LABEL_43:
    sub_CBEFB0(a1);
    return v65;
  }
  return result;
}
