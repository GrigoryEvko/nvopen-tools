// Function: sub_1244A70
// Address: 0x1244a70
//
__int64 __fastcall sub_1244A70(
        __int64 a1,
        char *a2,
        unsigned int a3,
        unsigned __int64 a4,
        int a5,
        char a6,
        int a7,
        int a8,
        char a9,
        char a10,
        char a11)
{
  const char *v15; // rax
  unsigned int v16; // r13d
  unsigned __int64 v18; // rax
  unsigned int v19; // eax
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r12
  _QWORD *v26; // rax
  unsigned __int64 v27; // r12
  char v28; // cl
  int v29; // ebx
  char v30; // cl
  char v31; // al
  char v32; // dl
  __int64 v33; // rsi
  __int64 v34; // rdi
  int v35; // eax
  __int64 v36; // rdx
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rsi
  char *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned int v46; // eax
  __int64 v47; // rax
  char v48; // dl
  __int64 **v49; // rsi
  __int64 *v50; // rax
  __int64 v51; // rax
  unsigned __int64 *v52; // rax
  unsigned __int64 *v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rax
  unsigned __int64 v57; // rsi
  unsigned __int64 v58; // [rsp+8h] [rbp-128h]
  char v59; // [rsp+1Bh] [rbp-115h]
  unsigned int v60; // [rsp+20h] [rbp-110h]
  char v63; // [rsp+3Bh] [rbp-F5h] BYREF
  int v64; // [rsp+3Ch] [rbp-F4h] BYREF
  __int64 *v65; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v66; // [rsp+48h] [rbp-E8h] BYREF
  unsigned __int64 v67; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v68; // [rsp+58h] [rbp-D8h] BYREF
  unsigned __int64 *v69[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v70; // [rsp+70h] [rbp-C0h] BYREF
  __m128i *v71; // [rsp+80h] [rbp-B0h] BYREF
  char *v72; // [rsp+88h] [rbp-A8h]
  __m128i v73; // [rsp+90h] [rbp-A0h] BYREF
  const char *v74; // [rsp+A0h] [rbp-90h] BYREF
  char *v75; // [rsp+A8h] [rbp-88h]
  __int64 v76; // [rsp+B0h] [rbp-80h]
  char v77; // [rsp+B8h] [rbp-78h] BYREF
  __int16 v78; // [rsp+C0h] [rbp-70h]

  v60 = a5 - 7;
  if ( (unsigned int)(a5 - 7) <= 1 )
  {
    if ( a7 )
    {
      HIBYTE(v78) = 1;
      v15 = "symbol with local linkage must have default visibility";
      goto LABEL_14;
    }
    if ( (a5 == 7 || a5 == 8) && a8 )
    {
      HIBYTE(v78) = 1;
      v15 = "symbol with local linkage cannot have a DLL storage class";
LABEL_14:
      LOBYTE(v78) = 3;
      v74 = v15;
      sub_11FD800(a1 + 176, a4, (__int64)&v74, 1);
      return 1;
    }
  }
  v65 = 0;
  v59 = sub_1212650(a1, &v64, 0);
  if ( v59 )
    return 1;
  if ( *(_DWORD *)(a1 + 240) == 44 )
  {
    v59 = 1;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  if ( (unsigned __int8)sub_120A640(a1, &v63) )
    return 1;
  v18 = *(_QWORD *)(a1 + 232);
  v78 = 259;
  v58 = v18;
  v74 = "expected type";
  if ( (unsigned __int8)sub_12190A0(a1, &v65, (int *)&v74, 0) )
    return 1;
  v66 = 0;
  if ( !a6 || a5 && a5 != 9 )
  {
    if ( (unsigned __int8)sub_1224770((__int64 **)a1, (__int64)v65, &v66) )
      return 1;
  }
  if ( *((_BYTE *)v65 + 8) == 13 || (LOBYTE(v19) = sub_BCBD20((__int64)v65), v16 = v19, !(_BYTE)v19) )
  {
    v78 = 259;
    v16 = 1;
    v74 = "invalid type for global variable";
    sub_11FD800(a1 + 176, v58, (__int64)&v74, 1);
    return v16;
  }
  v20 = *((_QWORD *)a2 + 1);
  if ( v20 )
  {
    v21 = sub_1212F00(a1 + 1096, (__int64)a2);
    if ( v21 == a1 + 1104 )
    {
      v22 = sub_BA8B30(*(_QWORD *)(a1 + 344), *(_QWORD *)a2, v20);
      if ( v22 )
      {
        sub_8FD6D0((__int64)v69, "redefinition of global '@", a2);
        if ( v69[1] == (unsigned __int64 *)0x3FFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"basic_string::append");
        v56 = sub_2241490(v69, "'", 1, v55);
        v71 = &v73;
        if ( *(_QWORD *)v56 == v56 + 16 )
        {
          v73 = _mm_loadu_si128((const __m128i *)(v56 + 16));
        }
        else
        {
          v71 = *(__m128i **)v56;
          v73.m128i_i64[0] = *(_QWORD *)(v56 + 16);
        }
        v72 = *(char **)(v56 + 8);
        *(_QWORD *)v56 = v56 + 16;
        *(_QWORD *)(v56 + 8) = 0;
        *(_BYTE *)(v56 + 16) = 0;
        v78 = 260;
        v74 = (const char *)&v71;
        sub_11FD800(a1 + 176, a4, (__int64)&v74, 1);
        if ( v71 != &v73 )
          j_j___libc_free_0(v71, v73.m128i_i64[0] + 1);
        if ( (__int64 *)v69[0] != &v70 )
          j_j___libc_free_0(v69[0], v70 + 1);
        return v16;
      }
    }
    else
    {
      v22 = *(_QWORD *)(v21 + 64);
      v23 = sub_220F330(v21, a1 + 1104);
      v24 = *(_QWORD *)(v23 + 32);
      v25 = v23;
      if ( v24 != v23 + 48 )
        j_j___libc_free_0(v24, *(_QWORD *)(v23 + 48) + 1LL);
      j_j___libc_free_0(v25, 80);
      --*(_QWORD *)(a1 + 1136);
    }
  }
  else
  {
    if ( a3 == -1 )
      a3 = *(_DWORD *)(a1 + 1224);
    v22 = *(_QWORD *)(a1 + 1160);
    v33 = a1 + 1152;
    if ( v22 )
    {
      v34 = a1 + 1152;
      do
      {
        if ( *(_DWORD *)(v22 + 32) < a3 )
        {
          v22 = *(_QWORD *)(v22 + 24);
        }
        else
        {
          v34 = v22;
          v22 = *(_QWORD *)(v22 + 16);
        }
      }
      while ( v22 );
      if ( v33 != v34 && *(_DWORD *)(v34 + 32) <= a3 )
      {
        v22 = *(_QWORD *)(v34 + 40);
        v47 = sub_220F330(v34, v33);
        j_j___libc_free_0(v47, 56);
        --*(_QWORD *)(a1 + 1184);
      }
    }
  }
  BYTE4(v71) = 1;
  v78 = 260;
  v74 = a2;
  LODWORD(v71) = v64;
  v26 = sub_BD2C40(88, unk_3F0FAE8);
  v27 = (unsigned __int64)v26;
  if ( v26 )
    sub_B30000((__int64)v26, *(_QWORD *)(a1 + 344), v65, 0, 0, 0, (__int64)&v74, 0, 0, (__int64)v71, 0);
  if ( !*((_QWORD *)a2 + 1) )
    sub_1243C70(a1 + 1192, a3, v27);
  if ( v66 )
    sub_B30160(v27, v66);
  v28 = a5;
  v29 = a5 & 0xF;
  v30 = v28 & 0xF;
  *(_BYTE *)(v27 + 80) = v63 & 1 | *(_BYTE *)(v27 + 80) & 0xFE;
  v31 = a7 & 3;
  if ( v60 <= 1 )
  {
    *(_WORD *)(v27 + 32) = v29 | *(_WORD *)(v27 + 32) & 0xFCC0;
    goto LABEL_34;
  }
  v48 = v30 | *(_BYTE *)(v27 + 32) & 0xF0;
  *(_BYTE *)(v27 + 32) = v48;
  if ( (unsigned int)(v29 - 7) <= 1 || (v48 & 0x30) != 0 && v30 != 9 )
  {
LABEL_34:
    v32 = *(_BYTE *)(v27 + 33) | 0x40;
    *(_BYTE *)(v27 + 33) = v32;
    if ( !a9 )
      goto LABEL_36;
    goto LABEL_35;
  }
  if ( a9 )
  {
    v32 = *(_BYTE *)(v27 + 33);
LABEL_35:
    v32 |= 0x40u;
    *(_BYTE *)(v27 + 33) = v32;
LABEL_36:
    *(_BYTE *)(v27 + 32) = (16 * v31) | *(_BYTE *)(v27 + 32) & 0xCF;
    if ( v29 == 7 )
      goto LABEL_41;
    goto LABEL_37;
  }
  *(_BYTE *)(v27 + 32) = (16 * v31) | *(_BYTE *)(v27 + 32) & 0xCF;
LABEL_37:
  if ( v29 != 8 && ((*(_BYTE *)(v27 + 32) & 0x30) == 0 || v30 == 9) )
    goto LABEL_42;
  v32 = *(_BYTE *)(v27 + 33);
LABEL_41:
  *(_BYTE *)(v27 + 33) = v32 | 0x40;
LABEL_42:
  *(_BYTE *)(v27 + 80) = (2 * (v59 & 1)) | *(_BYTE *)(v27 + 80) & 0xFD;
  *(_WORD *)(v27 + 32) = *(_WORD *)(v27 + 32) & 0xE03F | ((a10 & 7) << 10) | ((a11 & 3) << 6) | ((a8 & 3) << 8);
  if ( v22 )
  {
    if ( v64 != *(_DWORD *)(*(_QWORD *)(v22 + 8) + 8LL) >> 8 )
    {
      v74 = "forward reference and definition of global have different types";
      v78 = 259;
      sub_11FD800(a1 + 176, v58, (__int64)&v74, 1);
      return v16;
    }
    sub_BD84D0(v22, v27);
    sub_B30810((_QWORD *)v22);
  }
  if ( *(_DWORD *)(a1 + 240) == 4 )
  {
    while ( 1 )
    {
      v35 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v35;
      if ( v35 == 95 )
        break;
      switch ( v35 )
      {
        case 96:
          v43 = sub_1205200(a1 + 176);
          v44 = *(_QWORD *)(a1 + 256);
          v45 = *(_QWORD *)(a1 + 248);
          *(_DWORD *)(a1 + 240) = v43;
          sub_B30D10(v27, v45, v44);
          v40 = "expected partition string";
LABEL_69:
          if ( (unsigned __int8)sub_120AFE0(a1, 512, v40) )
            return 1;
          break;
        case 251:
          LOWORD(v74) = 0;
          v46 = sub_120CD10(a1, &v74, 0);
          if ( (_BYTE)v46 )
            return v46;
          if ( BYTE1(v74) )
            sub_B2F770(v27, (unsigned __int8)v74);
          break;
        case 97:
          v46 = sub_120CEA0(a1, &v74);
          if ( (_BYTE)v46 )
            return v46;
          sub_B30310(v27, (int)v74);
          break;
        case 511:
          if ( (unsigned __int8)sub_122E8D0(a1, v27) )
            return 1;
          break;
        default:
          if ( v35 == 223 || (unsigned int)(v35 - 499) <= 2 )
          {
            if ( (unsigned __int8)sub_120A860(a1, v27, v36) )
              return 1;
          }
          else
          {
            if ( (unsigned __int8)sub_121D100(a1, *(_BYTE **)a2, *((_QWORD *)a2 + 1), (__int64 *)&v71) )
              return (unsigned __int8)v16;
            if ( !v71 )
            {
              v57 = *(_QWORD *)(a1 + 232);
              v16 = (unsigned __int8)v16;
              v74 = "unknown global variable property!";
              v78 = 259;
              sub_11FD800(a1 + 176, v57, (__int64)&v74, 1);
              return v16;
            }
            sub_B2F990(v27, (__int64)v71, v41, v42);
          }
          break;
      }
      if ( *(_DWORD *)(a1 + 240) != 4 )
        goto LABEL_90;
    }
    v37 = sub_1205200(a1 + 176);
    v38 = *(_QWORD *)(a1 + 256);
    v39 = *(_QWORD *)(a1 + 248);
    *(_DWORD *)(a1 + 240) = v37;
    sub_B31A00(v27, v39, v38);
    v40 = "expected global section string";
    goto LABEL_69;
  }
LABEL_90:
  v49 = (__int64 **)&v74;
  v50 = **(__int64 ***)(a1 + 344);
  v75 = &v77;
  v67 = 0;
  v74 = (const char *)v50;
  v76 = 0x800000000LL;
  v71 = 0;
  v72 = 0;
  v73.m128i_i64[0] = 0;
  v16 = sub_1218010(a1, (__int64 **)&v74, (__int64)&v71, 0, &v67);
  if ( !(_BYTE)v16 && ((_DWORD)v76 || v72 != (char *)v71) )
  {
    v51 = sub_A7A280(*(__int64 **)a1, (__int64)&v74);
    v68 = v27;
    *(_QWORD *)(v27 + 72) = v51;
    v52 = *(unsigned __int64 **)(a1 + 1448);
    v53 = (unsigned __int64 *)(a1 + 1440);
    if ( !v52 )
      goto LABEL_101;
    do
    {
      if ( v52[4] < v27 )
      {
        v52 = (unsigned __int64 *)v52[3];
      }
      else
      {
        v53 = v52;
        v52 = (unsigned __int64 *)v52[2];
      }
    }
    while ( v52 );
    if ( v53 == (unsigned __int64 *)(a1 + 1440) || v53[4] > v27 )
    {
LABEL_101:
      v69[0] = &v68;
      v53 = sub_121BC10((_QWORD *)(a1 + 1432), v53, v69);
    }
    v54 = (__int64)(v53 + 5);
    v49 = (__int64 **)&v71;
    sub_1205F70(v54, (char **)&v71);
  }
  if ( v71 )
  {
    v49 = (__int64 **)(v73.m128i_i64[0] - (_QWORD)v71);
    j_j___libc_free_0(v71, v73.m128i_i64[0] - (_QWORD)v71);
  }
  if ( v75 != &v77 )
    _libc_free(v75, v49);
  return v16;
}
