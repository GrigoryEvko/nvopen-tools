// Function: sub_25B41B0
// Address: 0x25b41b0
//
void __fastcall sub_25B41B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r13
  __int64 p_src; // rsi
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rdx
  char v7; // al
  char *v8; // rax
  char *v9; // rdx
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // rcx
  unsigned __int64 *i; // rcx
  __int64 j; // r12
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rbx
  unsigned __int64 *v18; // r12
  __int64 v19; // rcx
  __int64 v20; // rbx
  unsigned __int8 *v21; // r12
  int v22; // eax
  unsigned __int64 v23; // rax
  __int64 v24; // rcx
  bool v25; // al
  bool v26; // cl
  __int64 k; // r12
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r15
  __int32 v31; // r13d
  const void *v32; // r8
  unsigned __int64 *v33; // r14
  __int64 v34; // rdi
  __int64 v35; // r9
  __int64 v36; // r12
  __int64 v37; // rcx
  int v38; // edx
  __int64 v39; // rbx
  __int64 v40; // r14
  __int32 v41; // r12d
  __int64 v42; // r14
  int v43; // edx
  __int64 v44; // r15
  int v45; // eax
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rbx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  char *v53; // rax
  char *v54; // rdx
  unsigned __int64 v55; // [rsp+20h] [rbp-310h]
  char v56; // [rsp+2Bh] [rbp-305h]
  unsigned int v57; // [rsp+2Ch] [rbp-304h]
  __int64 v58; // [rsp+30h] [rbp-300h]
  _QWORD *v59; // [rsp+38h] [rbp-2F8h]
  char v61; // [rsp+53h] [rbp-2DDh]
  unsigned int v62; // [rsp+54h] [rbp-2DCh]
  __int64 v63; // [rsp+58h] [rbp-2D8h]
  __int64 v64; // [rsp+60h] [rbp-2D0h]
  __int64 v65; // [rsp+68h] [rbp-2C8h]
  __m128i v66; // [rsp+70h] [rbp-2C0h] BYREF
  void *s; // [rsp+80h] [rbp-2B0h] BYREF
  __int64 v68; // [rsp+88h] [rbp-2A8h]
  _BYTE v69[4]; // [rsp+90h] [rbp-2A0h] BYREF
  char v70; // [rsp+94h] [rbp-29Ch] BYREF
  __m128i src; // [rsp+B0h] [rbp-280h] BYREF
  _BYTE v72[80]; // [rsp+C0h] [rbp-270h] BYREF
  unsigned __int64 *v73; // [rsp+110h] [rbp-220h] BYREF
  __int64 v74; // [rsp+118h] [rbp-218h]
  _BYTE v75[528]; // [rsp+120h] [rbp-210h] BYREF

  v2 = a1;
  src.m128i_i64[0] = *(_QWORD *)(a2 + 120);
  if ( (unsigned __int8)sub_A74390(src.m128i_i64, 83, 0)
    || (v73 = *(unsigned __int64 **)(a2 + 120), (unsigned __int8)sub_A74390((__int64 *)&v73, 84, 0))
    || (p_src = 20, (v61 = sub_B2D610(a2, 20)) != 0) )
  {
    sub_25B0AB0(a1, a2);
    return;
  }
  v6 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  v7 = *(_BYTE *)(v6 + 8);
  switch ( v7 )
  {
    case 7:
      HIDWORD(v68) = 5;
      s = v69;
      goto LABEL_94;
    case 15:
      v62 = *(_DWORD *)(v6 + 12);
      break;
    case 16:
      v62 = *(_DWORD *)(v6 + 32);
      break;
    default:
      v8 = v69;
      HIDWORD(v68) = 5;
      v9 = &v70;
      s = v69;
      v55 = 1;
      v62 = 1;
      goto LABEL_11;
  }
  v55 = v62;
  s = v69;
  v68 = 0x500000000LL;
  if ( v62 <= 5 )
  {
    if ( v62 )
    {
      v8 = v69;
      v9 = &v69[4 * v62];
      do
      {
LABEL_11:
        *(_DWORD *)v8 = 1;
        v8 += 4;
      }
      while ( v8 != v9 );
      v74 = 0x500000000LL;
      LODWORD(v68) = v62;
      v10 = (unsigned __int64 *)v75;
      v73 = (unsigned __int64 *)v75;
      v11 = (unsigned __int64 *)v75;
      goto LABEL_13;
    }
LABEL_94:
    LODWORD(v68) = 0;
    v73 = (unsigned __int64 *)v75;
    v74 = 0x500000000LL;
    v62 = 0;
    v55 = 0;
    goto LABEL_18;
  }
  sub_C8D5F0((__int64)&s, v69, v62, 4u, v4, v5);
  v53 = (char *)s;
  v54 = (char *)s + 4 * v62;
  do
  {
    *(_DWORD *)v53 = 1;
    v53 += 4;
  }
  while ( v54 != v53 );
  LODWORD(v68) = v62;
  v73 = (unsigned __int64 *)v75;
  v74 = 0x500000000LL;
  sub_25B3FF0((__int64)&v73, v62, (__int64)v54, v50, v51, v52);
  v11 = v73;
  v10 = &v73[12 * (unsigned int)v74];
LABEL_13:
  p_src = v55;
  v6 = 96 * v55;
  for ( i = &v11[12 * v55]; i != v10; v10 += 12 )
  {
    if ( v10 )
    {
      v6 = (unsigned __int64)(v10 + 2);
      *((_DWORD *)v10 + 2) = 0;
      *v10 = (unsigned __int64)(v10 + 2);
      *((_DWORD *)v10 + 3) = 5;
    }
  }
  LODWORD(v74) = v62;
LABEL_18:
  v56 = 0;
  for ( j = *(_QWORD *)(a2 + 80); a2 + 72 != j; j = *(_QWORD *)(j + 8) )
  {
    v14 = j - 24;
    if ( !j )
      v14 = 0;
    v15 = sub_AA4E50(v14);
    if ( v15 )
    {
      v16 = *(_QWORD *)(v15 - 32);
      if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(v15 + 80) || sub_B2FC80(v16) )
        goto LABEL_25;
      v56 = 1;
    }
  }
  v19 = a2;
  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 || *((_BYTE *)v2 + 144) && (*(_BYTE *)(a2 + 33) & 0x20) == 0 )
  {
    v58 = *(_QWORD *)(a2 + 16);
    if ( v58 )
    {
      v59 = v2;
      v57 = 0;
      v20 = 4LL * (v62 - 1) + 4;
      while ( 1 )
      {
        v21 = *(unsigned __int8 **)(v58 + 24);
        v22 = *v21;
        if ( (unsigned __int8)v22 <= 0x1Cu )
          break;
        v23 = (unsigned int)(v22 - 34);
        if ( (unsigned __int8)v23 > 0x33u )
          break;
        v24 = 0x8000000000041LL;
        if ( !_bittest64(&v24, v23)
          || (unsigned __int8 *)v58 != v21 - 32
          || *(_QWORD *)(a2 + 24) != *((_QWORD *)v21 + 10) )
        {
          break;
        }
        v25 = sub_B49200(*(_QWORD *)(v58 + 24));
        v26 = v61;
        if ( v25 )
          v26 = v25;
        v61 = v26;
        v19 = v62;
        if ( v57 != v62 )
        {
          for ( k = *((_QWORD *)v21 + 2); k; k = *(_QWORD *)(k + 8) )
          {
            p_src = *(_QWORD *)(k + 24);
            if ( *(_BYTE *)p_src == 93 )
            {
              v44 = **(unsigned int **)(p_src + 72);
              v19 = *((unsigned int *)s + v44);
              if ( (_DWORD)v19 )
              {
                v45 = sub_25B02B0(v59, p_src, (__int64)&v73[12 * v44]);
                v6 = (unsigned __int64)s;
                *((_DWORD *)s + v44) = v45;
                v57 += *((_DWORD *)s + v44) == 0;
              }
            }
            else
            {
              p_src = k;
              src.m128i_i64[0] = (__int64)v72;
              src.m128i_i64[1] = 0x500000000LL;
              if ( !(unsigned int)sub_25AFFC0(v59, k, (__int64)&src, 0xFFFFFFFF) )
              {
                if ( HIDWORD(v68) < v55 )
                {
                  LODWORD(v68) = 0;
                  sub_C8D5F0((__int64)&s, v69, v55, 4u, v28, v29);
                  p_src = 0;
                  memset(s, 0, 4 * v55);
                  LODWORD(v68) = v62;
                }
                else
                {
                  v46 = (unsigned int)v68;
                  v19 = v55;
                  v6 = v55;
                  if ( (unsigned int)v68 <= v55 )
                    v6 = (unsigned int)v68;
                  if ( v6 )
                  {
                    p_src = 0;
                    memset(s, 0, 4 * v6);
                    v46 = (unsigned int)v68;
                  }
                  if ( v46 < v55 )
                  {
                    v6 = v55 - v46;
                    if ( v55 != v46 )
                    {
                      v19 = (__int64)s;
                      v6 *= 4LL;
                      if ( v6 )
                      {
                        p_src = 0;
                        memset((char *)s + 4 * v46, 0, v6);
                      }
                    }
                  }
                  LODWORD(v68) = v62;
                }
                if ( (_BYTE *)src.m128i_i64[0] != v72 )
                  _libc_free(src.m128i_u64[0]);
                v57 = v62;
                break;
              }
              v6 = v62;
              v30 = 0;
              if ( v62 )
              {
                do
                {
                  while ( !*(_DWORD *)((char *)s + v30) )
                  {
                    v30 += 4;
                    if ( v20 == v30 )
                      goto LABEL_61;
                  }
                  v31 = src.m128i_i32[2];
                  v32 = (const void *)src.m128i_i64[0];
                  v33 = &v73[3 * v30];
                  v34 = *((unsigned int *)v33 + 2);
                  v35 = 16LL * src.m128i_u32[2];
                  v6 = src.m128i_u32[2] + v34;
                  if ( v6 > *((unsigned int *)v33 + 3) )
                  {
                    p_src = (__int64)(v33 + 2);
                    v63 = src.m128i_i64[0];
                    v64 = 16LL * src.m128i_u32[2];
                    sub_C8D5F0((__int64)&v73[3 * v30], v33 + 2, v6, 0x10u, src.m128i_i64[0], v35);
                    v34 = *((unsigned int *)v33 + 2);
                    v32 = (const void *)v63;
                    v35 = v64;
                  }
                  if ( v35 )
                  {
                    p_src = (__int64)v32;
                    memcpy((void *)(*v33 + 16 * v34), v32, v35);
                    LODWORD(v34) = *((_DWORD *)v33 + 2);
                  }
                  v30 += 4;
                  *((_DWORD *)v33 + 2) = v31 + v34;
                }
                while ( v20 != v30 );
              }
LABEL_61:
              if ( (_BYTE *)src.m128i_i64[0] != v72 )
                _libc_free(src.m128i_u64[0]);
            }
          }
        }
        v58 = *(_QWORD *)(v58 + 8);
        if ( !v58 )
        {
          v2 = v59;
          goto LABEL_66;
        }
      }
      v2 = v59;
      goto LABEL_25;
    }
LABEL_66:
    v36 = 0;
    if ( v62 )
    {
      do
      {
        p_src = (__int64)&src;
        v37 = (__int64)&v73[12 * v36];
        v38 = *((_DWORD *)s + v36);
        src.m128i_i32[2] = v36++;
        src.m128i_i64[0] = a2;
        src.m128i_i8[12] = 0;
        sub_25B0610(v2, &src, v38, v37);
      }
      while ( v62 != v36 );
    }
    src.m128i_i64[0] = (__int64)v72;
    src.m128i_i64[1] = 0x500000000LL;
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, p_src, v6, v19);
      v39 = *(_QWORD *)(a2 + 96);
      if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
        sub_B2C6D0(a2, p_src, v47, v48);
      v40 = *(_QWORD *)(a2 + 96) + 40LL * *(_QWORD *)(a2 + 104);
      if ( v39 == v40 )
        goto LABEL_105;
    }
    else
    {
      v39 = *(_QWORD *)(a2 + 96);
      v40 = v39 + 40LL * *(_QWORD *)(a2 + 104);
      if ( v40 == v39 )
      {
LABEL_107:
        v49 = (__int64)v73;
        v18 = &v73[12 * (unsigned int)v74];
        if ( v73 == v18 )
          goto LABEL_30;
        do
        {
          v18 -= 12;
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            _libc_free(*v18);
        }
        while ( (unsigned __int64 *)v49 != v18 );
        goto LABEL_29;
      }
    }
    v65 = v40;
    v41 = 0;
    v42 = v39;
    do
    {
      v43 = 0;
      if ( !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
      {
        if ( v56 || v61 )
          v43 = 0;
        else
          v43 = sub_25B02B0(v2, v42, (__int64)&src);
      }
      v66.m128i_i32[2] = v41;
      v66.m128i_i64[0] = a2;
      v42 += 40;
      ++v41;
      v66.m128i_i8[12] = 1;
      sub_25B0610(v2, &v66, v43, (__int64)&src);
      src.m128i_i32[2] = 0;
    }
    while ( v42 != v65 );
LABEL_105:
    if ( (_BYTE *)src.m128i_i64[0] != v72 )
      _libc_free(src.m128i_u64[0]);
    goto LABEL_107;
  }
LABEL_25:
  sub_25B0AB0(v2, a2);
  v17 = (__int64)v73;
  v18 = &v73[12 * (unsigned int)v74];
  if ( v73 != v18 )
  {
    do
    {
      v18 -= 12;
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        _libc_free(*v18);
    }
    while ( (unsigned __int64 *)v17 != v18 );
LABEL_29:
    v18 = v73;
  }
LABEL_30:
  if ( v18 != (unsigned __int64 *)v75 )
    _libc_free((unsigned __int64)v18);
  if ( s != v69 )
    _libc_free((unsigned __int64)s);
}
