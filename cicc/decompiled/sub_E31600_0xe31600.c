// Function: sub_E31600
// Address: 0xe31600
//
__int64 __fastcall sub_E31600(__int64 a1, __int64 a2, __int64 *a3)
{
  int v3; // r15d
  __int64 *v4; // r12
  __int64 v5; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 j; // rbx
  int v11; // eax
  unsigned int v12; // edx
  __int64 v13; // rsi
  char *v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned __int64 v20; // r15
  __int64 *v21; // r8
  unsigned __int64 v22; // r10
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r11
  unsigned __int64 v25; // rcx
  int v26; // esi
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // kr10_8
  __int64 v31; // rsi
  __int64 v32; // r11
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  int v37; // ecx
  unsigned __int16 v38; // cx
  __int64 v39; // r12
  unsigned __int64 v40; // rax
  char *v41; // rdi
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  int v45; // ecx
  __int64 v46; // rdx
  char *v47; // rdi
  _BYTE *v48; // rax
  _BYTE *v49; // rsi
  __int64 v50; // rcx
  _BYTE *v51; // rcx
  _BYTE *i; // rdx
  __int64 v53; // rcx
  __int64 *v54; // [rsp+8h] [rbp-68h]
  __int64 *v55; // [rsp+8h] [rbp-68h]
  unsigned __int64 v56; // [rsp+10h] [rbp-60h]
  int v57; // [rsp+10h] [rbp-60h]
  int v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  unsigned __int64 v62; // [rsp+28h] [rbp-48h]
  unsigned __int64 v63; // [rsp+28h] [rbp-48h]
  int v64; // [rsp+30h] [rbp-40h]

  v4 = a3;
  v60 = a3[1];
  if ( !a1 )
  {
    v46 = a3[1];
    goto LABEL_55;
  }
  v5 = a1;
  v7 = 0;
  v8 = -1;
  do
  {
    if ( *(_BYTE *)(a2 + v7) == 95 )
      v8 = v7;
    ++v7;
  }
  while ( v7 != a1 );
  if ( v8 == -1 )
  {
    v19 = 0;
    goto LABEL_20;
  }
  if ( !v8 )
  {
    v19 = 1;
LABEL_19:
    if ( v5 == v19 )
    {
      v46 = v4[1];
      goto LABEL_55;
    }
LABEL_20:
    v56 = 700;
    v62 = 0;
    v20 = 72;
    v21 = v4;
    v59 = 128;
    do
    {
      v22 = v62;
      v23 = 36;
      v24 = 36 - v20;
      v25 = 1;
      while ( 1 )
      {
        ++v19;
        v26 = *(char *)(a2 + v19 - 1);
        if ( (unsigned __int8)(*(_BYTE *)(a2 + v19 - 1) - 97) <= 0x19u )
        {
          v28 = v26 - 97;
        }
        else
        {
          if ( (unsigned __int8)(*(_BYTE *)(a2 + v19 - 1) - 48) > 9u )
            return 0;
          v28 = v26 - 22;
        }
        if ( ~v22 / v25 < v28 )
          return 0;
        v22 += v25 * v28;
        v29 = 1;
        if ( v20 < v23 )
        {
          v29 = 26;
          if ( v20 + 26 > v23 )
            v29 = v24;
        }
        if ( v29 > v28 )
          break;
        v30 = v25;
        v25 *= 36 - v29;
        if ( is_mul_ok(36 - v29, v30) )
        {
          v23 += 36LL;
          v24 += 36LL;
          if ( v19 != v5 )
            continue;
        }
        return 0;
      }
      v31 = v21[1];
      v32 = 0;
      v33 = ((unsigned __int64)(v31 - v60) >> 2) + 1;
      v34 = (v22 - v62) / v56 + (v22 - v62) / v56 / v33;
      if ( v34 > 0x1C7 )
      {
        do
        {
          v35 = v34;
          v32 += 36;
          v34 /= 0x23u;
        }
        while ( v35 > 0x3E57 );
      }
      v20 = 36 * v34 / (v34 + 38) + v32;
      v63 = v22 % v33;
      if ( v22 / v33 > ~v59 )
        return 0;
      v36 = v22 / v33 + v59;
      v37 = 0;
      v59 = v36;
      if ( v36 - 55296 <= 0x7FF )
        return 0;
      if ( v36 <= 0x7F )
      {
        LOBYTE(v37) = v36;
      }
      else if ( v36 <= 0x7FF )
      {
        LOBYTE(v37) = (v36 >> 6) | 0xC0;
        BYTE1(v37) = v36 & 0x3F | 0x80;
      }
      else if ( v36 <= 0xFFFF )
      {
        LOBYTE(v37) = (v36 >> 12) | 0xE0;
        BYTE1(v37) = (v36 >> 6) & 0x3F | 0x80;
        v37 = ((v36 & 0x3F | 0x80) << 16) | v37 & 0xFF00FFFF;
      }
      else
      {
        if ( v36 > 0x10FFFF )
          return 0;
        LOBYTE(v38) = (v36 >> 18) | 0xF0;
        HIBYTE(v38) = (v36 >> 12) & 0x3F | 0x80;
        v37 = ((v36 & 0x3F | 0x80) << 24) | v38 | (((v36 >> 6) & 0x3F | 0x80) << 16);
      }
      v39 = v60 + 4 * v63;
      v40 = v21[2];
      v41 = (char *)*v21;
      if ( v31 + 4 > v40 )
      {
        v42 = v31 + 996;
        v43 = 2 * v40;
        if ( v42 > v43 )
          v21[2] = v42;
        else
          v21[2] = v43;
        v54 = v21;
        v57 = v37;
        v44 = realloc(v41);
        v21 = v54;
        v41 = (char *)v44;
        *v54 = v44;
        if ( !v44 )
          goto LABEL_88;
        v31 = v54[1];
        v37 = v57;
      }
      v55 = v21;
      v58 = v37;
      memmove(&v41[v39 + 4], &v41[v39], v31 - v39);
      v21 = v55;
      v45 = v58;
      v56 = 2;
      *(_DWORD *)(*v55 + v39) = v45;
      v46 = v55[1] + 4;
      v55[1] = v46;
      v62 = v63 + 1;
    }
    while ( v19 != v5 );
    v4 = v55;
LABEL_55:
    v47 = (char *)*v4;
    v48 = (_BYTE *)(*v4 + v60);
    v49 = (_BYTE *)(*v4 + v46);
    v50 = (v46 - v60) >> 2;
    if ( v50 > 0 )
    {
      v51 = &v48[4 * v50];
      while ( *v48 )
      {
        if ( !v48[1] )
        {
          ++v48;
          goto LABEL_62;
        }
        if ( !v48[2] )
        {
          v48 += 2;
          goto LABEL_62;
        }
        if ( !v48[3] )
        {
          v48 += 3;
          goto LABEL_62;
        }
        v48 += 4;
        if ( v51 == v48 )
          goto LABEL_74;
      }
      goto LABEL_62;
    }
LABEL_74:
    v53 = v49 - v48;
    if ( v49 - v48 != 2 )
    {
      if ( v53 != 3 )
      {
        if ( v53 != 1 )
          goto LABEL_68;
        goto LABEL_77;
      }
      if ( !*v48 )
        goto LABEL_62;
      ++v48;
    }
    if ( !*v48 )
      goto LABEL_62;
    ++v48;
LABEL_77:
    if ( !*v48 )
    {
LABEL_62:
      if ( v49 == v48 )
      {
        v46 = v49 - v47;
      }
      else
      {
        for ( i = v48 + 1; v49 != i; ++i )
        {
          if ( *i )
            *v48++ = *i;
        }
        v46 = v48 - v47;
      }
    }
LABEL_68:
    v4[1] = v46;
    return 1;
  }
  v9 = v8;
  for ( j = 0; ; ++j )
  {
    v11 = *(unsigned __int8 *)(a2 + j);
    if ( (unsigned __int8)(v11 - 48) > 9u )
    {
      v12 = (v11 & 0xFFFFFFDF) - 65;
      LOBYTE(v12) = (_BYTE)v11 == 95 || (unsigned __int8)((v11 & 0xDF) - 65) <= 0x19u;
      if ( !(_BYTE)v12 )
        break;
    }
    BYTE2(v64) = 0;
    LOWORD(v64) = 0;
    v13 = v4[1];
    v14 = (char *)*v4;
    v3 = (v64 << 8) | (unsigned __int8)v3;
    LOBYTE(v3) = *(_BYTE *)(a2 + j);
    v15 = v4[2];
    if ( v13 + 4 > v15 )
    {
      v16 = v13 + 996;
      v17 = 2 * v15;
      if ( v16 > v17 )
        v4[2] = v16;
      else
        v4[2] = v17;
      v18 = realloc(v14);
      *v4 = v18;
      v14 = (char *)v18;
      if ( !v18 )
LABEL_88:
        abort();
      v13 = v4[1];
    }
    *(_DWORD *)&v14[v13] = v3;
    v4[1] += 4;
    if ( v9 == j + 1 )
    {
      v5 = a1;
      v19 = j + 2;
      goto LABEL_19;
    }
  }
  return v12;
}
