// Function: sub_1703010
// Address: 0x1703010
//
__int64 __fastcall sub_1703010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v6; // r14
  _BYTE *v7; // r12
  _BYTE *v8; // rbx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  unsigned int v14; // ecx
  __int64 v15; // rdx
  __int64 v16; // r13
  _QWORD *v17; // rax
  bool v18; // zf
  int v19; // eax
  __int64 v20; // r8
  unsigned __int8 v21; // dl
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rsi
  __int64 v26; // r11
  int v27; // eax
  unsigned int v28; // ecx
  _QWORD *v29; // rdi
  unsigned int v30; // eax
  int v31; // eax
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r13d
  __int64 v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *i; // rdx
  unsigned int v40; // r13d
  int v42; // r8d
  __int64 *v43; // r10
  __int64 v44; // rax
  const void *v45; // rsi
  __int64 *v46; // rbx
  __int64 v47; // r15
  __int64 *v48; // r14
  __int64 v49; // r12
  __int64 v50; // r15
  __m128i *v51; // rsi
  __int8 *v52; // rsi
  int v53; // esi
  int v54; // r15d
  _QWORD *v55; // rax
  _BYTE *v56; // [rsp+18h] [rbp-148h]
  __int64 v57; // [rsp+18h] [rbp-148h]
  __int64 v58; // [rsp+20h] [rbp-140h]
  __int64 v59; // [rsp+30h] [rbp-130h] BYREF
  int v60; // [rsp+38h] [rbp-128h] BYREF
  __m128i v61; // [rsp+40h] [rbp-120h] BYREF
  __int64 v62; // [rsp+50h] [rbp-110h]
  __int64 *v63; // [rsp+60h] [rbp-100h] BYREF
  __int64 v64; // [rsp+68h] [rbp-F8h]
  _QWORD v65[2]; // [rsp+70h] [rbp-F0h] BYREF
  char v66; // [rsp+80h] [rbp-E0h]
  _BYTE *v67; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v68; // [rsp+98h] [rbp-C8h]
  _BYTE v69[64]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE *v70; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v71; // [rsp+E8h] [rbp-78h]
  _BYTE v72[112]; // [rsp+F0h] [rbp-70h] BYREF

  v6 = a1;
  v7 = v72;
  v8 = v69;
  v68 = 0x800000000LL;
  v71 = 0x800000000LL;
  v58 = a1 + 80;
  v9 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 80);
  v67 = v69;
  v70 = v72;
  if ( v9 )
  {
    v28 = 4 * v9;
    v11 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)(4 * v9) < 0x40 )
      v28 = 64;
    if ( v28 >= (unsigned int)v11 )
      goto LABEL_4;
    v29 = *(_QWORD **)(a1 + 88);
    v30 = v9 - 1;
    if ( v30 )
    {
      _BitScanReverse(&v30, v30);
      v31 = 1 << (33 - (v30 ^ 0x1F));
      if ( v31 < 64 )
        v31 = 64;
      if ( (_DWORD)v11 == v31 )
      {
        *(_QWORD *)(v6 + 96) = 0;
        v55 = &v29[2 * (unsigned int)v11];
        do
        {
          if ( v29 )
            *v29 = -8;
          v29 += 2;
        }
        while ( v55 != v29 );
        goto LABEL_37;
      }
      v32 = (4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1);
      v33 = ((v32 | (v32 >> 2)) >> 4) | v32 | (v32 >> 2) | ((((v32 | (v32 >> 2)) >> 4) | v32 | (v32 >> 2)) >> 8);
      v34 = ((v33 >> 16) | v33) + 1;
      v35 = v34;
      v36 = 16 * v34;
    }
    else
    {
      v36 = 2048;
      v35 = 128;
    }
    j___libc_free_0(v29);
    *(_DWORD *)(v6 + 104) = v35;
    v37 = (_QWORD *)sub_22077B0(v36);
    v38 = *(unsigned int *)(v6 + 104);
    *(_QWORD *)(v6 + 96) = 0;
    *(_QWORD *)(v6 + 88) = v37;
    for ( i = &v37[2 * v38]; i != v37; v37 += 2 )
    {
      if ( v37 )
        *v37 = -8;
    }
LABEL_37:
    v10 = (unsigned int)v68;
    v14 = HIDWORD(v68);
    goto LABEL_8;
  }
  v10 = *(unsigned int *)(a1 + 100);
  if ( (_DWORD)v10 )
  {
    v11 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)v11 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 88));
      *(_QWORD *)(a1 + 88) = 0;
      v10 = (unsigned int)v68;
      *(_QWORD *)(a1 + 96) = 0;
      v14 = HIDWORD(v68);
      *(_DWORD *)(a1 + 104) = 0;
      goto LABEL_8;
    }
LABEL_4:
    v12 = *(_QWORD **)(a1 + 88);
    v13 = &v12[2 * v11];
    if ( v12 == v13 )
    {
      v14 = 8;
      v10 = 0;
    }
    else
    {
      do
      {
        *v12 = -8;
        v12 += 2;
      }
      while ( v13 != v12 );
      v10 = (unsigned int)v68;
      v14 = HIDWORD(v68);
    }
    *(_QWORD *)(a1 + 96) = 0;
LABEL_8:
    v15 = *(_QWORD *)(v6 + 112);
    if ( v15 == *(_QWORD *)(v6 + 120) )
      goto LABEL_10;
    goto LABEL_9;
  }
  v15 = *(_QWORD *)(a1 + 112);
  v14 = 8;
  if ( v15 == *(_QWORD *)(a1 + 120) )
  {
    v16 = *(_QWORD *)(*(_QWORD *)(a1 + 72) - 24LL);
    v17 = v69;
    goto LABEL_12;
  }
LABEL_9:
  *(_QWORD *)(v6 + 120) = v15;
LABEL_10:
  v16 = *(_QWORD *)(*(_QWORD *)(v6 + 72) - 24LL);
  if ( (unsigned int)v10 >= v14 )
  {
    sub_16CD150((__int64)&v67, v69, 0, 8, a5, a6);
    v17 = &v67[8 * (unsigned int)v68];
  }
  else
  {
    v17 = &v67[8 * v10];
  }
LABEL_12:
  *v17 = v16;
  v18 = (_DWORD)v68 == -1;
  v19 = v68 + 1;
  LODWORD(v68) = v68 + 1;
  if ( !v18 )
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)&v67[8 * v19 - 8];
      v21 = *(_BYTE *)(v20 + 16);
      if ( v21 <= 0x10u )
        goto LABEL_14;
      if ( v21 <= 0x17u )
      {
LABEL_38:
        v40 = 0;
        goto LABEL_39;
      }
      v22 = (unsigned int)v71;
      if ( (_DWORD)v71 && v20 == *(_QWORD *)&v70[8 * (unsigned int)v71 - 8] )
      {
        LODWORD(v71) = v71 - 1;
        LODWORD(v68) = v19 - 1;
        v61 = (__m128i)(unsigned __int64)v20;
        v62 = 0;
        v59 = v20;
        v60 = 0;
        sub_1702D60((__int64)&v63, v58, &v59, &v60);
        if ( v66 )
        {
          v50 = v65[0];
          v51 = *(__m128i **)(v6 + 120);
          if ( v51 == *(__m128i **)(v6 + 128) )
          {
            sub_1702BB0((const __m128i **)(v6 + 112), v51, &v61);
            v52 = *(__int8 **)(v6 + 120);
          }
          else
          {
            if ( v51 )
            {
              *v51 = _mm_loadu_si128(&v61);
              v51[1].m128i_i64[0] = v62;
              v51 = *(__m128i **)(v6 + 120);
            }
            v52 = &v51[1].m128i_i8[8];
            *(_QWORD *)(v6 + 120) = v52;
          }
          *(_DWORD *)(v50 + 8) = -1431655765 * ((__int64)&v52[-*(_QWORD *)(v6 + 112)] >> 3) - 1;
        }
LABEL_52:
        v19 = v68;
        if ( !(_DWORD)v68 )
          break;
      }
      else
      {
        v23 = *(unsigned int *)(v6 + 104);
        if ( !(_DWORD)v23 )
          goto LABEL_22;
        a6 = *(_QWORD *)(v6 + 88);
        v24 = (v23 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v25 = (__int64 *)(a6 + 16LL * v24);
        v26 = *v25;
        if ( v20 != *v25 )
        {
          v53 = 1;
          while ( v26 != -8 )
          {
            v54 = v53 + 1;
            v24 = (v23 - 1) & (v53 + v24);
            v25 = (__int64 *)(a6 + 16LL * v24);
            v26 = *v25;
            if ( v20 == *v25 )
              goto LABEL_21;
            v53 = v54;
          }
LABEL_22:
          if ( (unsigned int)v71 >= HIDWORD(v71) )
          {
            v57 = *(_QWORD *)&v67[8 * v19 - 8];
            sub_16CD150((__int64)&v70, v7, 0, 8, v20, a6);
            v22 = (unsigned int)v71;
            v20 = v57;
          }
          *(_QWORD *)&v70[8 * v22] = v20;
          v27 = *(unsigned __int8 *)(v20 + 16);
          LODWORD(v71) = v71 + 1;
          switch ( v27 )
          {
            case '#':
            case '%':
            case '\'':
            case '2':
            case '3':
            case '4':
              v64 = 0x200000000LL;
              v63 = v65;
              sub_1702010(v20, (__int64)&v63, v23, v22, v20, a6);
              v42 = (int)v63;
              v43 = &v63[(unsigned int)v64];
              if ( v63 != v43 )
              {
                v56 = v7;
                v44 = (unsigned int)v68;
                v45 = v8;
                v46 = v63;
                v47 = v6;
                v48 = &v63[(unsigned int)v64];
                do
                {
                  v49 = *v46;
                  if ( HIDWORD(v68) <= (unsigned int)v44 )
                  {
                    sub_16CD150((__int64)&v67, v45, 0, 8, v42, a6);
                    v44 = (unsigned int)v68;
                  }
                  ++v46;
                  *(_QWORD *)&v67[8 * v44] = v49;
                  v44 = (unsigned int)(v68 + 1);
                  LODWORD(v68) = v68 + 1;
                }
                while ( v48 != v46 );
                v6 = v47;
                v7 = v56;
                v8 = v45;
                v43 = v63;
              }
              if ( v43 != v65 )
                _libc_free((unsigned __int64)v43);
              goto LABEL_52;
            case '<':
            case '=':
            case '>':
              goto LABEL_52;
            default:
              goto LABEL_38;
          }
        }
LABEL_21:
        v23 = a6 + 16 * v23;
        if ( v25 == (__int64 *)v23 )
          goto LABEL_22;
LABEL_14:
        LODWORD(v68) = --v19;
        if ( !v19 )
          break;
      }
    }
  }
  v40 = 1;
LABEL_39:
  if ( v70 != v7 )
    _libc_free((unsigned __int64)v70);
  if ( v67 != v8 )
    _libc_free((unsigned __int64)v67);
  return v40;
}
