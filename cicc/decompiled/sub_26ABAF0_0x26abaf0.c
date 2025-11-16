// Function: sub_26ABAF0
// Address: 0x26abaf0
//
unsigned __int64 __fastcall sub_26ABAF0(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdi
  __m128i v4; // xmm2
  __m128i v5; // xmm3
  void (__fastcall *v6)(_BYTE *, _BYTE *, __int64); // rcx
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r13
  unsigned __int64 v10; // rdx
  __int64 v11; // r8
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v16; // rax
  int v17; // eax
  int v18; // edx
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  int v24; // r14d
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r13
  __int64 v31; // rbx
  __int64 v32; // rdx
  __int64 v33; // rsi
  int v34; // eax
  unsigned int v35; // ecx
  int v36; // eax
  __int64 v37; // rax
  int v38; // edx
  bool v39; // zf
  int v40; // edx
  unsigned __int8 v41; // si
  __int64 *v42; // rax
  bool v43; // cc
  unsigned __int64 v44; // rax
  __int128 v45; // [rsp-18h] [rbp-268h]
  __int64 v46; // [rsp+8h] [rbp-248h]
  __int64 v47; // [rsp+10h] [rbp-240h]
  int v49; // [rsp+28h] [rbp-228h]
  int v50; // [rsp+2Ch] [rbp-224h]
  unsigned __int64 v51; // [rsp+38h] [rbp-218h]
  int v52; // [rsp+40h] [rbp-210h]
  int v53; // [rsp+44h] [rbp-20Ch]
  unsigned __int64 v54; // [rsp+48h] [rbp-208h]
  __m128i *v55; // [rsp+50h] [rbp-200h] BYREF
  __int64 v56; // [rsp+58h] [rbp-1F8h]
  _BYTE v57[32]; // [rsp+60h] [rbp-1F0h] BYREF
  __m128i v58; // [rsp+80h] [rbp-1D0h]
  __m128i v59; // [rsp+90h] [rbp-1C0h]
  _BYTE v60[16]; // [rsp+A0h] [rbp-1B0h] BYREF
  void (__fastcall *v61)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-1A0h]
  unsigned __int8 (__fastcall *v62)(_BYTE *, __int64); // [rsp+B8h] [rbp-198h]
  __m128i v63; // [rsp+C0h] [rbp-190h]
  __m128i v64; // [rsp+D0h] [rbp-180h]
  _BYTE v65[16]; // [rsp+E0h] [rbp-170h] BYREF
  void (__fastcall *v66)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-160h]
  __int64 v67; // [rsp+F8h] [rbp-158h]
  __m128i v68; // [rsp+100h] [rbp-150h] BYREF
  __m128i v69; // [rsp+110h] [rbp-140h] BYREF
  _BYTE v70[16]; // [rsp+120h] [rbp-130h] BYREF
  void (__fastcall *v71)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-120h]
  unsigned __int8 (__fastcall *v72)(_BYTE *, __int64); // [rsp+138h] [rbp-118h]
  __m128i v73; // [rsp+140h] [rbp-110h] BYREF
  __m128i v74; // [rsp+150h] [rbp-100h] BYREF
  _BYTE v75[16]; // [rsp+160h] [rbp-F0h] BYREF
  void (__fastcall *v76)(_BYTE *, _BYTE *, __int64); // [rsp+170h] [rbp-E0h]
  __int64 v77; // [rsp+178h] [rbp-D8h]
  _BYTE v78[24]; // [rsp+180h] [rbp-D0h] BYREF
  char *v79; // [rsp+198h] [rbp-B8h]
  char v80; // [rsp+1A8h] [rbp-A8h] BYREF
  char *v81; // [rsp+1C8h] [rbp-88h]
  char v82; // [rsp+1D8h] [rbp-78h] BYREF

  v47 = sub_AA4E30(a1);
  v3 = &v68;
  v49 = sub_30D4FD0();
  sub_AA72C0(&v68, a1, 1);
  v61 = 0;
  v58 = _mm_loadu_si128(&v68);
  v59 = _mm_loadu_si128(&v69);
  if ( v71 )
  {
    v3 = (__m128i *)v60;
    v71(v60, v70, 2);
    v62 = v72;
    v61 = v71;
  }
  v4 = _mm_loadu_si128(&v73);
  v5 = _mm_loadu_si128(&v74);
  v66 = 0;
  v63 = v4;
  v64 = v5;
  if ( v76 )
  {
    v3 = (__m128i *)v65;
    v76(v65, v75, 2);
    v6 = v76;
    v67 = v77;
    v7 = v58.m128i_i64[0];
    v66 = v76;
    v8 = v58.m128i_i64[0];
    if ( v58.m128i_i64[0] == v63.m128i_i64[0] )
    {
      v54 = 0;
      goto LABEL_26;
    }
  }
  else
  {
    v7 = v58.m128i_i64[0];
    v8 = v58.m128i_i64[0];
    if ( v58.m128i_i64[0] == v63.m128i_i64[0] )
    {
      v54 = 0;
      goto LABEL_28;
    }
  }
  v54 = 0;
  v53 = 0;
  do
  {
    if ( !v8 )
      BUG();
    v9 = v8 - 24;
    v10 = (unsigned int)*(unsigned __int8 *)(v8 - 24) - 29;
    if ( *(_BYTE *)(v8 - 24) == 63 )
    {
      v3 = (__m128i *)(v8 - 24);
      if ( (unsigned __int8)sub_B4DCF0(v8 - 24) )
      {
        v7 = v58.m128i_i64[0];
        goto LABEL_16;
      }
    }
    else if ( (unsigned int)v10 <= 0x22 )
    {
      if ( *(_BYTE *)(v8 - 24) == 60 )
        goto LABEL_16;
    }
    else
    {
      if ( (unsigned int)v10 > 0x31 )
      {
        if ( *(_BYTE *)(v8 - 24) == 84 )
          goto LABEL_16;
        v3 = (__m128i *)(v8 - 24);
        if ( sub_B46A10(v8 - 24) )
        {
LABEL_37:
          v7 = v58.m128i_i64[0];
          goto LABEL_16;
        }
        goto LABEL_12;
      }
      if ( (unsigned int)v10 > 0x2E )
        goto LABEL_16;
    }
    v3 = (__m128i *)(v8 - 24);
    if ( sub_B46A10(v8 - 24) )
      goto LABEL_37;
LABEL_12:
    v12 = *(_BYTE *)(v8 - 24);
    if ( v12 != 85 )
    {
      if ( v12 == 34 )
      {
LABEL_14:
        v3 = (__m128i *)a2;
        v13 = (int)sub_30D4FE0(a2, v8 - 24, v47);
        v10 = (int)v13 + v54;
        if ( !__OFADD__((int)v13, v54) )
          goto LABEL_15;
      }
      else
      {
        if ( v12 == 32 )
        {
          v20 = v49 * ((*(_DWORD *)(v8 - 20) & 0x7FFFFFFu) >> 1);
          v10 = v20 + v54;
          if ( !__OFADD__(v20, v54) )
            goto LABEL_15;
          if ( !(v49 * ((*(_DWORD *)(v8 - 20) & 0x7FFFFFFu) >> 1)) )
          {
LABEL_56:
            v54 = 0x8000000000000000LL;
            v7 = v58.m128i_i64[0];
            goto LABEL_16;
          }
          goto LABEL_106;
        }
        v13 = v49;
        v10 = v49 + v54;
        if ( !__OFADD__(v49, v54) )
        {
LABEL_15:
          v54 = v10;
          v7 = v58.m128i_i64[0];
          goto LABEL_16;
        }
      }
      if ( v13 <= 0 )
        goto LABEL_56;
LABEL_106:
      v54 = 0x7FFFFFFFFFFFFFFFLL;
      v7 = v58.m128i_i64[0];
      goto LABEL_16;
    }
    v16 = *(_QWORD *)(v8 - 56);
    if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(v8 + 56) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
      goto LABEL_14;
    v17 = *(_DWORD *)(v16 + 36);
    v52 = 0;
    v55 = (__m128i *)v57;
    v50 = v17;
    v56 = 0x400000000LL;
    v18 = *(unsigned __int8 *)(v8 - 24);
    if ( v18 == 40 )
    {
      v19 = -32 - 32LL * (unsigned int)sub_B491D0(v8 - 24);
    }
    else
    {
      v19 = -32;
      if ( v18 != 85 )
      {
        if ( v18 != 34 )
          BUG();
        v19 = -96;
      }
    }
    if ( *(char *)(v8 - 17) < 0 )
    {
      v21 = sub_BD2BC0(v8 - 24);
      v23 = v21 + v22;
      if ( *(char *)(v8 - 17) >= 0 )
      {
        if ( (unsigned int)(v23 >> 4) )
LABEL_114:
          BUG();
      }
      else if ( (unsigned int)((v23 - sub_BD2BC0(v8 - 24)) >> 4) )
      {
        if ( *(char *)(v8 - 17) >= 0 )
          goto LABEL_114;
        v24 = *(_DWORD *)(sub_BD2BC0(v8 - 24) + 8);
        if ( *(char *)(v8 - 17) >= 0 )
          BUG();
        v25 = sub_BD2BC0(v8 - 24);
        v19 -= 32LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v24);
      }
    }
    v27 = v9 + v19;
    v28 = (unsigned int)v56;
    v29 = v27;
    v30 = v9 - 32LL * (*(_DWORD *)(v8 - 20) & 0x7FFFFFF);
    if ( v30 != v27 )
    {
      do
      {
        v31 = *(_QWORD *)(*(_QWORD *)v30 + 8LL);
        if ( v28 + 1 > (unsigned __int64)HIDWORD(v56) )
        {
          sub_C8D5F0((__int64)&v55, v57, v28 + 1, 8u, v11, v27);
          v28 = (unsigned int)v56;
        }
        v30 += 32;
        v55->m128i_i64[v28] = v31;
        v28 = (unsigned int)(v56 + 1);
        LODWORD(v56) = v56 + 1;
      }
      while ( v29 != v30 );
    }
    v32 = *(_QWORD *)(v8 - 16);
    if ( *(_BYTE *)(v8 - 24) > 0x1Cu )
    {
      switch ( *(_BYTE *)(v8 - 24) )
      {
        case ')':
        case '+':
        case '-':
        case '/':
        case '2':
        case '5':
        case 'J':
        case 'K':
        case 'S':
          goto LABEL_78;
        case 'T':
        case 'U':
        case 'V':
          v34 = *(unsigned __int8 *)(v32 + 8);
          v35 = v34 - 17;
          v41 = *(_BYTE *)(v32 + 8);
          if ( (unsigned int)(v34 - 17) <= 1 )
            v41 = *(_BYTE *)(**(_QWORD **)(v32 + 16) + 8LL);
          if ( v41 <= 3u || v41 == 5 || (v41 & 0xFD) == 4 )
            goto LABEL_78;
          if ( (_BYTE)v34 == 15 )
          {
            if ( (*(_BYTE *)(v32 + 9) & 4) == 0 )
              break;
            v46 = *(_QWORD *)(v8 - 16);
            if ( !sub_BCB420(v46) )
            {
              v32 = *(_QWORD *)(v8 - 16);
              break;
            }
            v42 = *(__int64 **)(v46 + 16);
            v32 = *(_QWORD *)(v8 - 16);
            v33 = *v42;
            v34 = *(unsigned __int8 *)(*v42 + 8);
            v35 = v34 - 17;
          }
          else
          {
            v33 = *(_QWORD *)(v8 - 16);
            if ( (_BYTE)v34 == 16 )
            {
              do
              {
                v33 = *(_QWORD *)(v33 + 24);
                LOBYTE(v34) = *(_BYTE *)(v33 + 8);
              }
              while ( (_BYTE)v34 == 16 );
              v35 = (unsigned __int8)v34 - 17;
            }
          }
          if ( v35 <= 1 )
            LOBYTE(v34) = *(_BYTE *)(**(_QWORD **)(v33 + 16) + 8LL);
          if ( (unsigned __int8)v34 <= 3u || (_BYTE)v34 == 5 || (v34 & 0xFD) == 4 )
          {
LABEL_78:
            v36 = -1;
            if ( *(_BYTE *)(v8 - 23) >> 1 != 127 )
              v36 = *(_BYTE *)(v8 - 23) >> 1;
            v52 = v36;
          }
          break;
        default:
          break;
      }
    }
    v51 = v51 & 0xFFFFFFFF00000000LL | 1;
    *((_QWORD *)&v45 + 1) = v51;
    *(_QWORD *)&v45 = 0;
    sub_DF8CB0((__int64)v78, v50, v32, v55->m128i_i8, (unsigned int)v56, v52, 0, v45);
    v37 = sub_DFD690(a2, (__int64)v78);
    v39 = v38 == 1;
    v40 = 1;
    if ( !v39 )
      v40 = v53;
    v53 = v40;
    v10 = v37 + v54;
    if ( __OFADD__(v37, v54) )
    {
      v43 = v37 <= 0;
      v44 = 0x8000000000000000LL;
      if ( !v43 )
        v44 = 0x7FFFFFFFFFFFFFFFLL;
      v54 = v44;
    }
    else
    {
      v54 += v37;
    }
    if ( v81 != &v82 )
      _libc_free((unsigned __int64)v81);
    if ( v79 != &v80 )
      _libc_free((unsigned __int64)v79);
    v3 = v55;
    if ( v55 == (__m128i *)v57 )
      goto LABEL_37;
    _libc_free((unsigned __int64)v55);
    v7 = v58.m128i_i64[0];
LABEL_16:
    v8 = *(_QWORD *)(v7 + 8);
    v58.m128i_i16[4] = 0;
    v58.m128i_i64[0] = v8;
    v7 = v8;
    if ( v8 != v59.m128i_i64[0] )
    {
      v14 = v8;
      do
      {
        if ( v14 )
          v14 -= 24;
        if ( !v61 )
          sub_4263D6(v3, v14, v10);
        v3 = (__m128i *)v60;
        if ( v62(v60, v14) )
        {
          v8 = v58.m128i_i64[0];
          v7 = v58.m128i_i64[0];
          goto LABEL_24;
        }
        v10 = 0;
        v14 = *(_QWORD *)(v58.m128i_i64[0] + 8);
        v58.m128i_i16[4] = 0;
        v58.m128i_i64[0] = v14;
        v7 = v14;
      }
      while ( v59.m128i_i64[0] != v14 );
      v8 = v14;
    }
LABEL_24:
    ;
  }
  while ( v63.m128i_i64[0] != v8 );
  v6 = v66;
LABEL_26:
  if ( v6 )
    v6(v65, v65, 3);
LABEL_28:
  if ( v61 )
    v61(v60, v60, 3);
  if ( v76 )
    v76(v75, v75, 3);
  if ( v71 )
    v71(v70, v70, 3);
  return v54;
}
