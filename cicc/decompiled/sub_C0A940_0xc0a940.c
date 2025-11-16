// Function: sub_C0A940
// Address: 0xc0a940
//
__int64 __fastcall sub_C0A940(__int64 a1, char *a2, char *a3, __int64 a4)
{
  char v9; // al
  char *v10; // rax
  _BYTE *v11; // rdx
  __int64 v12; // rax
  _BYTE *v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rsi
  int v19; // eax
  __int64 v20; // rax
  _BYTE *v21; // rdx
  unsigned __int64 v22; // rdx
  unsigned __int8 v23; // al
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 *v28; // rax
  unsigned __int8 *v29; // rax
  unsigned __int8 *v30; // rdx
  unsigned __int64 v31; // rax
  __int64 *v32; // rdi
  _DWORD *v33; // r15
  _DWORD *v34; // rbx
  char v35; // al
  __int64 v36; // rax
  int v37; // edx
  unsigned __int64 v38; // rax
  char v39; // dl
  unsigned __int64 *v40; // rcx
  unsigned __int64 *v41; // r15
  __int64 *v42; // rbx
  char v43; // al
  __int64 v44; // rax
  int v45; // edx
  __m128i v46; // rax
  __int64 v47; // r12
  unsigned __int64 v48; // rdx
  __int64 *v49; // rax
  unsigned __int64 v50; // [rsp+0h] [rbp-2E0h]
  unsigned __int64 v51; // [rsp+0h] [rbp-2E0h]
  __int64 v52; // [rsp+0h] [rbp-2E0h]
  char v53; // [rsp+Bh] [rbp-2D5h]
  unsigned int v54; // [rsp+Ch] [rbp-2D4h]
  __int64 v55; // [rsp+18h] [rbp-2C8h]
  __int64 v56; // [rsp+18h] [rbp-2C8h]
  char s2; // [rsp+28h] [rbp-2B8h]
  char v58; // [rsp+3Bh] [rbp-2A5h]
  int v59; // [rsp+3Ch] [rbp-2A4h]
  __m128i v60; // [rsp+40h] [rbp-2A0h] BYREF
  unsigned __int64 v61; // [rsp+5Ch] [rbp-284h]
  __int64 v62; // [rsp+68h] [rbp-278h]
  int v63; // [rsp+70h] [rbp-270h]
  __int64 v64; // [rsp+74h] [rbp-26Ch]
  int v65; // [rsp+7Ch] [rbp-264h]
  void *s1[2]; // [rsp+80h] [rbp-260h] BYREF
  unsigned __int8 *v67[2]; // [rsp+90h] [rbp-250h] BYREF
  _BYTE *v68; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-238h]
  _BYTE v70[128]; // [rsp+B0h] [rbp-230h] BYREF
  unsigned __int64 v71; // [rsp+130h] [rbp-1B0h] BYREF
  _QWORD v72[2]; // [rsp+138h] [rbp-1A8h] BYREF
  char v73; // [rsp+148h] [rbp-198h] BYREF
  unsigned __int64 v74; // [rsp+1D0h] [rbp-110h] BYREF
  _QWORD v75[2]; // [rsp+1D8h] [rbp-108h] BYREF
  char v76; // [rsp+1E8h] [rbp-F8h] BYREF
  __int64 *v77; // [rsp+268h] [rbp-78h] BYREF
  __int64 v78; // [rsp+278h] [rbp-68h] BYREF
  __int64 *v79; // [rsp+288h] [rbp-58h] BYREF
  __int64 v80; // [rsp+298h] [rbp-48h] BYREF
  int v81; // [rsp+2A8h] [rbp-38h]

  v60.m128i_i64[0] = (__int64)a2;
  v60.m128i_i64[1] = (__int64)a3;
  *(__m128i *)s1 = _mm_load_si128(&v60);
  if ( (unsigned __int64)a3 <= 3 )
    goto LABEL_3;
  if ( *(_DWORD *)a2 != 1447516767 )
    goto LABEL_3;
  v60.m128i_i64[0] = (__int64)(a2 + 4);
  v60.m128i_i64[1] = (__int64)(a3 - 4);
  if ( a3 == (char *)4 )
    goto LABEL_3;
  if ( (unsigned __int64)(a3 - 4) > 5 && *((_DWORD *)a2 + 1) == 1447840863 && *((_WORD *)a2 + 4) == 24397 )
  {
    v10 = a3 - 10;
    v11 = a2 + 10;
    v59 = 7;
    v60.m128i_i64[0] = (__int64)(a2 + 10);
    v60.m128i_i64[1] = (__int64)(a3 - 10);
  }
  else
  {
    v9 = a2[4];
    if ( v9 == 110 )
    {
      v59 = 0;
    }
    else
    {
      v59 = 1;
      if ( v9 != 115 )
      {
        v59 = 2;
        if ( v9 != 114 )
        {
          v59 = 3;
          if ( v9 != 98 )
          {
            v59 = 4;
            if ( v9 != 99 )
            {
              v59 = 5;
              if ( v9 != 100 )
                v59 = 2 * (v9 != 101) + 6;
            }
          }
        }
      }
    }
    v10 = a3 - 5;
    v11 = a2 + 5;
    v60.m128i_i64[0] = (__int64)(a2 + 5);
    v60.m128i_i64[1] = (__int64)(a3 - 5);
  }
  if ( !v10 )
    goto LABEL_3;
  if ( *v11 == 77 )
  {
    v12 = (__int64)(v10 - 1);
    v13 = v11 + 1;
    v58 = 1;
    v60.m128i_i64[0] = (__int64)v13;
    v60.m128i_i64[1] = v12;
  }
  else
  {
    if ( *v11 != 78 )
      goto LABEL_3;
    v12 = (__int64)(v10 - 1);
    v13 = v11 + 1;
    v58 = 0;
    v60.m128i_i64[0] = (__int64)v13;
    v60.m128i_i64[1] = v12;
  }
  if ( !v12 || *v13 != 120 )
  {
    if ( !(unsigned __int8)sub_C93B20(&v60, 10, &v74) && v74 == (unsigned int)v74 )
    {
      v54 = v74;
      if ( v74 )
      {
        v53 = 0;
        v14 = v60.m128i_i64[1];
        v15 = (_BYTE *)v60.m128i_i64[0];
        goto LABEL_23;
      }
    }
LABEL_3:
    *(_BYTE *)(a1 + 224) = 0;
    return a1;
  }
  v14 = v12 - 1;
  v15 = v13 + 1;
  v60.m128i_i64[0] = (__int64)v15;
  v60.m128i_i64[1] = v14;
  if ( (unsigned int)(v59 - 1) > 1 )
    goto LABEL_3;
  v53 = 1;
  v54 = 0;
LABEL_23:
  v55 = a4;
  v16 = v50;
  v17 = 0;
  v68 = v70;
  v69 = 0x800000000LL;
  while ( v14 )
  {
    if ( *v15 == 118 )
    {
      v20 = v14 - 1;
      v21 = v15 + 1;
      LODWORD(v67[0]) = 0;
      v60.m128i_i64[0] = (__int64)v21;
      v60.m128i_i64[1] = v20;
      LODWORD(v71) = 0;
    }
    else
    {
      if ( *v15 != 117 )
        break;
      v20 = v14 - 1;
      v21 = v15 + 1;
      LODWORD(v67[0]) = 9;
      v60.m128i_i64[0] = (__int64)v21;
      v60.m128i_i64[1] = v20;
      LODWORD(v71) = 0;
    }
LABEL_40:
    if ( v20 && *v21 == 97 )
    {
      v18 = 10;
      v60.m128i_i64[0] = (__int64)(v21 + 1);
      v60.m128i_i64[1] = v20 - 1;
      if ( (unsigned __int8)sub_C93B20(&v60, 10, &v74) || !v74 || (v74 & (v74 - 1)) != 0 )
        goto LABEL_30;
      _BitScanReverse64(&v22, v74);
      v23 = 63 - (v22 ^ 0x3F);
    }
    else
    {
      v23 = 0;
    }
    v24 = v17 | ((unsigned __int64)LODWORD(v67[0]) << 32);
    v25 = ((unsigned __int64)v23 << 32) | (unsigned int)v71 | v16 & 0xFFFFFF0000000000LL;
    v26 = (unsigned int)v69;
    v16 = v25;
    v27 = (unsigned int)v69 + 1LL;
    if ( v27 > HIDWORD(v69) )
    {
      v51 = v24;
      sub_C8D5F0(&v68, v70, v27, 16);
      v26 = (unsigned int)v69;
      v24 = v51;
    }
    v28 = (unsigned __int64 *)&v68[16 * v26];
    *v28 = v24;
    v15 = (_BYTE *)v60.m128i_i64[0];
    v28[1] = v25;
    v17 = (unsigned int)(v69 + 1);
    v14 = v60.m128i_i64[1];
    LODWORD(v69) = v69 + 1;
  }
  v18 = (__int64)v67;
  v19 = sub_C0A100((char **)&v60, v67, &v71, byte_432C6C5, 2u);
  if ( v19 == 1 )
  {
    v18 = (__int64)v67;
    v19 = sub_C0A100((char **)&v60, v67, &v71, "Rs", 2u);
    if ( v19 == 1 )
    {
      v18 = (__int64)v67;
      v19 = sub_C0A100((char **)&v60, v67, &v71, "Ls", 2u);
      if ( v19 == 1 )
      {
        v18 = (__int64)v67;
        v19 = sub_C0A100((char **)&v60, v67, &v71, "Us", 2u);
        if ( v19 == 1 )
        {
          if ( (unsigned int)sub_C09FF0((__int64)&v60, v67, &v71, "l", 1u) )
          {
            if ( (unsigned int)sub_C09FF0((__int64)&v60, v67, &v71, "R", 1u) )
            {
              if ( (unsigned int)sub_C09FF0((__int64)&v60, v67, &v71, "L", 1u) )
              {
                v18 = (__int64)v67;
                if ( (unsigned int)sub_C09FF0((__int64)&v60, v67, &v71, "U", 1u) )
                  goto LABEL_28;
              }
            }
          }
          goto LABEL_62;
        }
      }
    }
  }
  if ( v19 == 2 )
    goto LABEL_30;
  if ( !v19 )
  {
LABEL_62:
    v20 = v60.m128i_i64[1];
    v21 = (_BYTE *)v60.m128i_i64[0];
    goto LABEL_40;
  }
LABEL_28:
  if ( !(_DWORD)v69 )
    goto LABEL_30;
  v18 = *(unsigned int *)(v55 + 12);
  if ( (_DWORD)v18 - 1 != (_DWORD)v69 )
    goto LABEL_30;
  if ( !v53 )
  {
    BYTE4(v61) = 0;
    s2 = 0;
    LODWORD(v61) = v54;
LABEL_66:
    if ( !v60.m128i_i64[1] )
      goto LABEL_30;
    if ( *(_BYTE *)v60.m128i_i64[0] != 95 )
      goto LABEL_30;
    ++v60.m128i_i64[0];
    --v60.m128i_i64[1];
    v29 = (unsigned __int8 *)sub_C09E90(
                               (char **)&v60,
                               (__int64 (__fastcall *)(__int64, _QWORD))sub_C09DD0,
                               (__int64)&v74);
    v67[0] = v29;
    v18 = (__int64)v29;
    v67[1] = v30;
    if ( !v30 )
      goto LABEL_30;
    v31 = sub_C935B0(&v60, v29, v30, 0);
    if ( v31 < v60.m128i_i64[1] )
    {
      v46.m128i_i64[1] = v60.m128i_i64[1] - v31;
      v46.m128i_i64[0] = v60.m128i_i64[0] + v31;
      v60 = v46;
      if ( *(_BYTE *)v46.m128i_i64[0] == 40 )
      {
        v18 = v46.m128i_i64[0] + 1;
        v60.m128i_i64[0] = v46.m128i_i64[0] + 1;
        v60.m128i_i64[1] = v46.m128i_i64[1] - 1;
        if ( v46.m128i_i64[1] == 1 )
          goto LABEL_30;
        if ( *(_BYTE *)(v46.m128i_i64[0] + v46.m128i_i64[1] - 1) != 41 )
          goto LABEL_30;
        v60.m128i_i64[1] = v46.m128i_i64[1] - 2;
        *(__m128i *)s1 = _mm_load_si128(&v60);
        if ( v46.m128i_i64[1] == 2 )
          goto LABEL_30;
      }
    }
    else
    {
      v60 = (__m128i)(unsigned __int64)(v60.m128i_i64[1] + v60.m128i_i64[0]);
    }
    if ( v59 == 7 && s1[1] == a3 )
    {
      v18 = (__int64)a2;
      if ( !memcmp(s1[0], a2, (size_t)a3) )
        goto LABEL_30;
    }
    if ( v58 )
    {
      v47 = (unsigned int)v69 | 0xA00000000LL;
      v48 = (unsigned int)v69 + 1LL;
      if ( v48 > HIDWORD(v69) )
        sub_C8D5F0(&v68, v70, v48, 16);
      v49 = (__int64 *)&v68[16 * (unsigned int)v69];
      *v49 = v47;
      v49[1] = 0;
      LODWORD(v69) = v69 + 1;
    }
    LODWORD(v61) = v54;
    BYTE4(v61) = s2;
    v71 = v61;
    sub_C0A690(v72, (__int64)&v68);
    v74 = v71;
    sub_C0A690(v75, (__int64)v72);
    sub_C0A5E0((__int64 *)&v77, v67);
    sub_C0A5E0((__int64 *)&v79, (unsigned __int8 **)s1);
    v18 = (__int64)&v74;
    v81 = v59;
    sub_C0A730(a1, (__int64)&v74);
    v32 = v79;
    *(_BYTE *)(a1 + 224) = 1;
    if ( v32 != &v80 )
    {
      v18 = v80 + 1;
      j_j___libc_free_0(v32, v80 + 1);
    }
    if ( v77 != &v78 )
    {
      v18 = v78 + 1;
      j_j___libc_free_0(v77, v78 + 1);
    }
    if ( (char *)v75[0] != &v76 )
      _libc_free(v75[0], v18);
    if ( (char *)v72[0] != &v73 )
      _libc_free(v72[0], v18);
    goto LABEL_31;
  }
  v33 = v68;
  s2 = 1;
  v54 = -1;
  v34 = &v68[16 * (unsigned int)(v18 - 1)];
  while ( 2 )
  {
    if ( !v33[1] )
    {
      v52 = *(_QWORD *)(*(_QWORD *)(v55 + 16) + 8LL * (unsigned int)(*v33 + 1));
      if ( sub_BCAC40(v52, 64) || (v35 = *(_BYTE *)(v52 + 8), v35 == 14) || v35 == 3 )
      {
        LODWORD(v36) = 2;
        v18 = 1;
      }
      else
      {
        v36 = sub_C09DE0(v52);
        v62 = v36;
        v18 = BYTE4(v36);
        v63 = v37;
        if ( !(_BYTE)v37 )
          goto LABEL_30;
        if ( !BYTE4(v36) )
        {
LABEL_89:
          if ( (unsigned int)v36 < v54 )
          {
            s2 = v18;
            v54 = v36;
          }
          goto LABEL_82;
        }
      }
      if ( s2 )
        goto LABEL_89;
    }
LABEL_82:
    v33 += 4;
    if ( v34 != v33 )
      continue;
    break;
  }
  v38 = **(_QWORD **)(v55 + 16);
  v74 = v38;
  v39 = *(_BYTE *)(v38 + 8);
  if ( v39 == 7 )
    goto LABEL_111;
  if ( v39 != 15 )
  {
    v41 = v75;
    v40 = &v74;
LABEL_99:
    v42 = (__int64 *)v40;
    while ( 1 )
    {
      v56 = *v42;
      if ( sub_BCAC40(*v42, 64) || (v43 = *(_BYTE *)(v56 + 8), v43 == 14) || v43 == 3 )
      {
        LODWORD(v44) = 2;
        v18 = 1;
      }
      else
      {
        v44 = sub_C09DE0(v56);
        v64 = v44;
        v18 = BYTE4(v44);
        v65 = v45;
        if ( !(_BYTE)v45 )
          goto LABEL_30;
        if ( !BYTE4(v44) )
        {
LABEL_104:
          if ( v54 > (unsigned int)v44 )
          {
            s2 = v18;
            v54 = v44;
          }
          goto LABEL_106;
        }
      }
      if ( s2 )
        goto LABEL_104;
LABEL_106:
      if ( v41 == (unsigned __int64 *)++v42 )
        goto LABEL_111;
    }
  }
  if ( (*(_DWORD *)(v38 + 8) & 0x400) != 0 && (*(_DWORD *)(v38 + 8) & 0x200) == 0 )
  {
    v40 = *(unsigned __int64 **)(v38 + 16);
    v41 = &v40[*(unsigned int *)(v38 + 12)];
    if ( v41 != v40 )
      goto LABEL_99;
LABEL_111:
    if ( v54 != -1 )
    {
      LODWORD(v61) = v54;
      BYTE4(v61) = s2;
      goto LABEL_66;
    }
  }
LABEL_30:
  *(_BYTE *)(a1 + 224) = 0;
LABEL_31:
  if ( v68 != v70 )
    _libc_free(v68, v18);
  return a1;
}
