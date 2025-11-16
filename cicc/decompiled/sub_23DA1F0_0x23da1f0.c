// Function: sub_23DA1F0
// Address: 0x23da1f0
//
__int64 __fastcall sub_23DA1F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rbx
  bool v15; // zf
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 v22; // r10
  unsigned __int8 *v23; // r12
  unsigned int v24; // r12d
  unsigned int v26; // ecx
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  int v29; // ebx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *i; // rdx
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  void *v37; // r10
  __int64 v38; // r8
  __int32 v39; // r12d
  unsigned __int64 v40; // rdx
  char *v41; // r8
  char *v42; // r10
  _QWORD *v43; // rax
  __int64 v44; // r12
  __int64 v45; // rdx
  _QWORD *v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rdx
  _QWORD *v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rdx
  unsigned int v52; // r11d
  _QWORD *v53; // rax
  __int64 v54; // [rsp+8h] [rbp-118h]
  char *v55; // [rsp+10h] [rbp-110h]
  __int64 v56; // [rsp+10h] [rbp-110h]
  char *v57; // [rsp+18h] [rbp-108h]
  void *v58; // [rsp+18h] [rbp-108h]
  __int64 v59; // [rsp+28h] [rbp-F8h]
  void *src; // [rsp+30h] [rbp-F0h] BYREF
  __m128i v61; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v62; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v63; // [rsp+58h] [rbp-C8h]
  _BYTE v64[64]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE *v65; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-78h]
  _BYTE v67[112]; // [rsp+B0h] [rbp-70h] BYREF

  v63 = 0x800000000LL;
  v66 = 0x800000000LL;
  v59 = a1 + 88;
  v7 = *(_DWORD *)(a1 + 104);
  ++*(_QWORD *)(a1 + 88);
  v62 = v64;
  v65 = v67;
  if ( !v7 )
  {
    if ( !*(_DWORD *)(a1 + 108) )
    {
      *(_DWORD *)(a1 + 128) = 0;
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 80) - 32LL);
      v11 = 0;
      goto LABEL_10;
    }
    v8 = *(unsigned int *)(a1 + 112);
    if ( (unsigned int)v8 <= 0x40 )
      goto LABEL_4;
    sub_C7D6A0(*(_QWORD *)(a1 + 96), 16LL * *(unsigned int *)(a1 + 112), 8);
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
    *(_DWORD *)(a1 + 112) = 0;
    goto LABEL_84;
  }
  v26 = 4 * v7;
  v8 = *(unsigned int *)(a1 + 112);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v26 = 64;
  if ( v26 < (unsigned int)v8 )
  {
    v27 = v7 - 1;
    if ( v27 )
    {
      _BitScanReverse(&v27, v27);
      v28 = *(_QWORD **)(a1 + 96);
      v29 = 1 << (33 - (v27 ^ 0x1F));
      if ( v29 < 64 )
        v29 = 64;
      if ( (_DWORD)v8 == v29 )
      {
        *(_QWORD *)(a1 + 104) = 0;
        v53 = &v28[2 * (unsigned int)v8];
        do
        {
          if ( v28 )
            *v28 = -4096;
          v28 += 2;
        }
        while ( v53 != v28 );
        goto LABEL_84;
      }
    }
    else
    {
      v28 = *(_QWORD **)(a1 + 96);
      v29 = 64;
    }
    sub_C7D6A0((__int64)v28, 16 * v8, 8);
    v30 = ((((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
             | (4 * v29 / 3u + 1)
             | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
           | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
         | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 16;
    v31 = (v30
         | (((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
             | (4 * v29 / 3u + 1)
             | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
           | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
           | (4 * v29 / 3u + 1)
           | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
         | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
         | (4 * v29 / 3u + 1)
         | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 112) = v31;
    v32 = (_QWORD *)sub_C7D670(16 * v31, 8);
    v33 = *(unsigned int *)(a1 + 112);
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 96) = v32;
    for ( i = &v32[2 * v33]; i != v32; v32 += 2 )
    {
      if ( v32 )
        *v32 = -4096;
    }
LABEL_84:
    v11 = (unsigned int)v63;
    v12 = HIDWORD(v63);
    v13 = (unsigned int)v63 + 1LL;
    goto LABEL_8;
  }
LABEL_4:
  v9 = *(_QWORD **)(a1 + 96);
  v10 = &v9[2 * v8];
  if ( v9 == v10 )
  {
    v12 = 8;
    v13 = 1;
    v11 = 0;
  }
  else
  {
    do
    {
      *v9 = -4096;
      v9 += 2;
    }
    while ( v10 != v9 );
    v11 = (unsigned int)v63;
    v12 = HIDWORD(v63);
    v13 = (unsigned int)v63 + 1LL;
  }
  *(_QWORD *)(a1 + 104) = 0;
LABEL_8:
  *(_DWORD *)(a1 + 128) = 0;
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 80) - 32LL);
  if ( v13 > v12 )
  {
    sub_C8D5F0((__int64)&v62, v64, v13, 8u, a5, a6);
    v11 = (unsigned int)v63;
  }
LABEL_10:
  *(_QWORD *)&v62[8 * v11] = v14;
  v15 = (_DWORD)v63 == -1;
  v16 = v63 + 1;
  LODWORD(v63) = v63 + 1;
  if ( !v15 )
  {
    do
    {
LABEL_19:
      v18 = (__int64)v62;
      v23 = *(unsigned __int8 **)&v62[8 * v16 - 8];
      if ( *v23 <= 0x15u )
      {
LABEL_17:
        LODWORD(v63) = --v16;
        continue;
      }
      if ( *v23 <= 0x1Cu )
      {
LABEL_21:
        v24 = 0;
        goto LABEL_22;
      }
      v17 = (unsigned int)v66;
      if ( (_DWORD)v66 )
      {
        v18 = (__int64)v65;
        if ( v23 == *(unsigned __int8 **)&v65[8 * (unsigned int)v66 - 8] )
        {
          src = *(void **)&v62[8 * v16 - 8];
          LODWORD(v66) = v66 - 1;
          LODWORD(v63) = v16 - 1;
          v61 = 0u;
          sub_23D9E70(v59, (__int64 *)&src, &v61);
          v16 = v63;
          continue;
        }
      }
      v19 = *(unsigned int *)(a1 + 112);
      v20 = *(_QWORD *)(a1 + 96);
      if ( (_DWORD)v19 )
      {
        a6 = (unsigned int)(v19 - 1);
        v21 = a6 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v18 = v20 + 16LL * v21;
        v22 = *(_QWORD *)v18;
        if ( v23 == *(unsigned __int8 **)v18 )
        {
LABEL_16:
          if ( v18 != v20 + 16 * v19 )
            goto LABEL_17;
        }
        else
        {
          v18 = 1;
          while ( v22 != -4096 )
          {
            v52 = v18 + 1;
            v21 = a6 & (v18 + v21);
            v18 = v20 + 16LL * v21;
            v22 = *(_QWORD *)v18;
            if ( v23 == *(unsigned __int8 **)v18 )
              goto LABEL_16;
            v18 = v52;
          }
        }
      }
      v35 = (unsigned int)v66 + 1LL;
      if ( v35 > HIDWORD(v66) )
      {
        sub_C8D5F0((__int64)&v65, v67, v35, 8u, v20, a6);
        v17 = (unsigned int)v66;
      }
      *(_QWORD *)&v65[8 * v17] = v23;
      LODWORD(v66) = v66 + 1;
      switch ( *v23 )
      {
        case '*':
        case ',':
        case '.':
        case '0':
        case '3':
        case '6':
        case '7':
        case '8':
        case '9':
        case ':':
        case ';':
        case 'V':
        case 'Z':
        case '[':
          src = &v61.m128i_u64[1];
          v61.m128i_i64[0] = 0x200000000LL;
          sub_23D8620(v23, (__int64)&src, v35, v18, v20, (__int64)&v61.m128i_i64[1]);
          v36 = (unsigned int)v63;
          v37 = src;
          v38 = 8LL * v61.m128i_u32[0];
          v39 = v61.m128i_i32[0];
          v40 = (unsigned int)v63 + (unsigned __int64)v61.m128i_u32[0];
          a6 = (__int64)&v61.m128i_i64[1];
          if ( v40 > HIDWORD(v63) )
          {
            v56 = 8LL * v61.m128i_u32[0];
            v58 = src;
            sub_C8D5F0((__int64)&v62, v64, v40, 8u, v38, (__int64)&v61.m128i_i64[1]);
            v36 = (unsigned int)v63;
            a6 = (__int64)&v61.m128i_i64[1];
            v38 = v56;
            v37 = v58;
          }
          if ( v38 )
          {
            memcpy(&v62[8 * v36], v37, v38);
            LODWORD(v36) = v63;
            a6 = (__int64)&v61.m128i_i64[1];
          }
          LODWORD(v63) = v36 + v39;
          v16 = v36 + v39;
          if ( src == &v61.m128i_u64[1] )
            continue;
          _libc_free((unsigned __int64)src);
          v16 = v63;
          if ( !(_DWORD)v63 )
            goto LABEL_50;
          goto LABEL_19;
        case 'C':
        case 'D':
        case 'E':
          goto LABEL_64;
        case 'T':
          src = &v61.m128i_u64[1];
          v61.m128i_i64[0] = 0x200000000LL;
          sub_23D8620(v23, (__int64)&src, v35, v18, v20, (__int64)&v61.m128i_i64[1]);
          v41 = (char *)src;
          a6 = (__int64)&v61.m128i_i64[1];
          v42 = (char *)src + 8 * v61.m128i_u32[0];
          if ( v42 == src )
            goto LABEL_62;
          break;
        default:
          goto LABEL_21;
      }
      do
      {
        v43 = v65;
        v44 = *(_QWORD *)v41;
        v45 = 8LL * (unsigned int)v66;
        v46 = &v65[v45];
        v47 = v45 >> 3;
        v48 = v45 >> 5;
        if ( v48 )
        {
          v49 = &v65[32 * v48];
          while ( v44 != *v43 )
          {
            if ( v44 == v43[1] )
            {
              ++v43;
              break;
            }
            if ( v44 == v43[2] )
            {
              v43 += 2;
              break;
            }
            if ( v44 == v43[3] )
            {
              v43 += 3;
              break;
            }
            v43 += 4;
            if ( v43 == v49 )
            {
              v47 = v46 - v43;
              goto LABEL_66;
            }
          }
LABEL_59:
          if ( v46 != v43 )
            goto LABEL_60;
          goto LABEL_69;
        }
LABEL_66:
        if ( v47 != 2 )
        {
          if ( v47 != 3 )
          {
            if ( v47 != 1 )
              goto LABEL_69;
            goto LABEL_76;
          }
          if ( v44 == *v43 )
            goto LABEL_59;
          ++v43;
        }
        if ( v44 == *v43 )
          goto LABEL_59;
        ++v43;
LABEL_76:
        if ( v44 == *v43 )
          goto LABEL_59;
LABEL_69:
        v50 = (unsigned int)v63;
        v51 = (unsigned int)v63 + 1LL;
        if ( v51 > HIDWORD(v63) )
        {
          v54 = a6;
          v55 = v41;
          v57 = v42;
          sub_C8D5F0((__int64)&v62, v64, v51, 8u, (__int64)v41, a6);
          v50 = (unsigned int)v63;
          a6 = v54;
          v41 = v55;
          v42 = v57;
        }
        *(_QWORD *)&v62[8 * v50] = v44;
        LODWORD(v63) = v63 + 1;
LABEL_60:
        v41 += 8;
      }
      while ( v42 != v41 );
      v41 = (char *)src;
LABEL_62:
      if ( v41 != (char *)a6 )
        _libc_free((unsigned __int64)v41);
LABEL_64:
      v16 = v63;
    }
    while ( v16 );
  }
LABEL_50:
  v24 = 1;
LABEL_22:
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  return v24;
}
