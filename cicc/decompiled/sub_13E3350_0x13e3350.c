// Function: sub_13E3350
// Address: 0x13e3350
//
__int64 __fastcall sub_13E3350(__int64 a1, const __m128i *a2, __int64 a3, char a4, __int64 a5)
{
  char v7; // dl
  int v9; // eax
  __int64 v10; // rdi
  __int64 **v11; // rax
  _QWORD *v12; // r13
  char v13; // al
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // rcx
  _QWORD *v19; // rax
  _QWORD *v20; // rcx
  char v21; // si
  _QWORD *v22; // rdx
  char v23; // cl
  __int64 *v24; // rax
  __int64 *v25; // rax
  char v26; // r13
  char v27; // dl
  __int64 *v28; // rax
  char v29; // dl
  __int64 v30; // rax
  char v31; // r13
  char v32; // dl
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 **v35; // r13
  __int64 **v36; // r8
  unsigned __int64 v37; // r9
  __int64 **v38; // rax
  int v39; // edx
  __int64 **v40; // rsi
  __int64 v41; // rdi
  char v42; // dl
  unsigned __int8 **v43; // rax
  __int64 v44; // rax
  char v45; // dl
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rax
  char v49; // dl
  __int64 v50; // rax
  char v51; // r13
  char v52; // dl
  __int64 v53; // rax
  unsigned __int8 **v54; // rax
  unsigned __int8 **v55; // rax
  __int64 v56; // rax
  char v57; // dl
  unsigned __int8 **v58; // rax
  char v59; // dl
  unsigned __int8 **v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rax
  char v63; // dl
  __int64 v64; // rax
  unsigned int v65; // r12d
  int v66; // r13d
  int v67; // eax
  __int64 **v68; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v69; // [rsp+8h] [rbp-B8h]
  __m128i v70; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v71; // [rsp+20h] [rbp-A0h]
  __int64 v72; // [rsp+30h] [rbp-90h]
  __int64 **v73; // [rsp+40h] [rbp-80h] BYREF
  __int64 v74; // [rsp+48h] [rbp-78h]
  __int64 v75; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v76; // [rsp+58h] [rbp-68h]

  v7 = a4;
  if ( a2[2].m128i_i64[0] )
    v72 = a2[2].m128i_i64[0];
  else
    v72 = a1;
  v70 = _mm_loadu_si128(a2);
  v71 = _mm_loadu_si128(a2 + 1);
  v9 = *(unsigned __int8 *)(a1 + 16);
  v10 = (unsigned int)(v9 - 24);
  switch ( v9 )
  {
    case '#':
      v26 = sub_15F2370(a1);
      v27 = sub_15F2380(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v28 = *(__int64 **)(a1 - 8);
      else
        v28 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13DEB20(*v28, v28[3], v27, v26, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '$':
      v42 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v43 = *(unsigned __int8 ***)(a1 - 8);
      else
        v43 = (unsigned __int8 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = (_QWORD *)sub_13D6AE0(*v43, v43[3], v42, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '%':
      v31 = sub_15F2370(a1);
      v32 = sub_15F2380(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v33 = *(_QWORD *)(a1 - 8);
      else
        v33 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = sub_13DF290(*(_QWORD **)v33, *(_QWORD *)(v33 + 24), v32, v31, v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '&':
      v29 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v30 = *(_QWORD *)(a1 - 8);
      else
        v30 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13D1D30(*(unsigned __int8 **)v30, *(_QWORD *)(v30 + 24), v29, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '\'':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v54 = *(unsigned __int8 ***)(a1 - 8);
      else
        v54 = (unsigned __int8 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13E06F0(*v54, v54[3], &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '(':
      v59 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v60 = *(unsigned __int8 ***)(a1 - 8);
      else
        v60 = (unsigned __int8 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = (_QWORD *)sub_13D1D40(*v60, v60[3], v59, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case ')':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v47 = *(_QWORD *)(a1 - 8);
      else
        v47 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13E11B0(*(_QWORD **)v47, *(_BYTE **)(v47 + 24), v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '*':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v62 = *(_QWORD *)(a1 - 8);
      else
        v62 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13E11D0(*(_QWORD **)v62, *(_BYTE **)(v62 + 24), v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '+':
      v49 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v50 = *(_QWORD *)(a1 - 8);
      else
        v50 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13D6F10(*(_QWORD **)v50, *(_QWORD *)(v50 + 24), v49, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case ',':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v56 = *(_QWORD *)(a1 - 8);
      else
        v56 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13E0AC0(*(_QWORD **)v56, *(unsigned __int8 **)(v56 + 24), v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '-':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v44 = *(_QWORD *)(a1 - 8);
      else
        v44 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13E0AB0(*(_QWORD **)v44, *(_QWORD *)(v44 + 24), v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '.':
      v63 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v64 = *(_QWORD *)(a1 - 8);
      else
        v64 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13D1D50(*(unsigned __int8 **)v64, *(_QWORD *)(v64 + 24), v63, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '/':
      v51 = sub_15F2370(a1);
      v52 = sub_15F2380(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v53 = *(_QWORD *)(a1 - 8);
      else
        v53 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = sub_13E10E0(*(unsigned __int8 **)v53, *(_QWORD *)(v53 + 24), v52, v51, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '0':
      v57 = sub_15F23D0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v58 = *(unsigned __int8 ***)(a1 - 8);
      else
        v58 = (unsigned __int8 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13E1070(*v58, v58[3], v57, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '1':
      v45 = sub_15F23D0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v46 = *(_QWORD *)(a1 - 8);
      else
        v46 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = sub_13E1000(*(unsigned __int8 **)v46, *(_QWORD *)(v46 + 24), v45, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '2':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v61 = *(__int64 **)(a1 - 8);
      else
        v61 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = (_QWORD *)sub_13E01B0(*v61, v61[3], &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '3':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v48 = *(__int64 **)(a1 - 8);
      else
        v48 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13E1270(*v48, v48[3], &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '4':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v55 = *(unsigned __int8 ***)(a1 - 8);
      else
        v55 = (unsigned __int8 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13DE4D0(*v55, v55[3], &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case '5':
      v12 = 0;
      goto LABEL_8;
    case '8':
      v34 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v35 = (__int64 **)(a1 - v34 * 8);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v35 = *(__int64 ***)(a1 - 8);
      v73 = (__int64 **)&v75;
      v36 = &v35[v34];
      v74 = 0x800000000LL;
      v37 = 0xAAAAAAAAAAAAAAABLL * ((v34 * 8) >> 3);
      if ( (unsigned __int64)v34 > 24 )
      {
        v68 = &v35[v34];
        v69 = 0xAAAAAAAAAAAAAAABLL * ((v34 * 8) >> 3);
        sub_16CD150(&v73, &v75, v69, 8);
        v40 = v73;
        v39 = v74;
        LODWORD(v37) = v69;
        v36 = v68;
        v38 = &v73[(unsigned int)v74];
      }
      else
      {
        v38 = (__int64 **)&v75;
        v39 = 0;
        v40 = (__int64 **)&v75;
      }
      if ( v36 != v35 )
      {
        do
        {
          if ( v38 )
            *v38 = *v35;
          v35 += 3;
          ++v38;
        }
        while ( v36 != v35 );
        v40 = v73;
        v39 = v74;
      }
      v41 = *(_QWORD *)(a1 + 56);
      LODWORD(v74) = v37 + v39;
      v12 = (_QWORD *)sub_13E3340(v41, v40, (unsigned int)(v37 + v39), v70.m128i_i64);
      if ( v73 != (__int64 **)&v75 )
        _libc_free((unsigned __int64)v73);
      goto LABEL_25;
    case '<':
    case '=':
    case '>':
    case '?':
    case '@':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v11 = *(__int64 ***)(a1 - 8);
      else
        v11 = (__int64 **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = (_QWORD *)sub_13D1870(v10, *v11, *(_QWORD *)a1, &v70, a5);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'K':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v25 = *(__int64 **)(a1 - 8);
      else
        v25 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13DD0E0(*(_WORD *)(a1 + 18) & 0x7FFF, *v25, v25[3], v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'L':
      v23 = sub_15F24E0(a1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v24 = *(__int64 **)(a1 - 8);
      else
        v24 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v12 = sub_13D91C0(*(_WORD *)(a1 + 18) & 0x7FFF, *v24, v24[3], v23, &v70);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'M':
      v18 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v19 = (_QWORD *)(a1 - v18 * 8);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v19 = *(_QWORD **)(a1 - 8);
      v20 = &v19[v18];
      if ( v20 == v19 )
        goto LABEL_158;
      v21 = 0;
      v12 = 0;
      break;
    case 'N':
      v16 = 0;
      v17 = *(_BYTE *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      if ( v17 > 0x17u )
      {
        if ( v17 == 78 )
        {
          v16 = a1 & 0xFFFFFFFFFFFFFFF8LL | 4;
        }
        else if ( v17 == 29 )
        {
          v16 = a1 & 0xFFFFFFFFFFFFFFF8LL;
        }
      }
      v12 = (_QWORD *)sub_13D3F90(v16, v70.m128i_i64);
LABEL_25:
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'O':
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v15 = *(_QWORD *)(a1 - 8);
      else
        v15 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v12 = (_QWORD *)sub_13E2B90(*(_QWORD *)v15, *(_QWORD *)(v15 + 24), *(_QWORD *)(v15 + 48), v70.m128i_i64);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'S':
      v12 = (_QWORD *)sub_13D1770(*(_BYTE **)(a1 - 48), *(_QWORD *)(a1 - 24));
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'T':
      v12 = (_QWORD *)sub_13D1600(*(__int64 **)(a1 - 72), *(_QWORD *)(a1 - 48), *(_QWORD *)(a1 - 24));
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'U':
      v12 = (_QWORD *)sub_13D1880(*(_BYTE **)(a1 - 72), *(_QWORD *)(a1 - 48), *(_BYTE **)(a1 - 24), *(_QWORD *)a1);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'V':
      v12 = (_QWORD *)sub_13D16C0(*(_QWORD *)(a1 - 24), *(const void **)(a1 + 56), *(_DWORD *)(a1 + 64));
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    case 'W':
      v12 = (_QWORD *)sub_13D1570(
                        *(_QWORD *)(a1 - 48),
                        *(_QWORD *)(a1 - 24),
                        *(const void **)(a1 + 56),
                        *(unsigned int *)(a1 + 64));
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
    default:
      v12 = (_QWORD *)sub_14DD210(a1, v70.m128i_i64[0], v70.m128i_i64[1]);
      v7 = a4 & (v12 == 0);
      goto LABEL_8;
  }
  do
  {
    v22 = (_QWORD *)*v19;
    if ( a1 != *v19 )
    {
      if ( *((_BYTE *)v22 + 16) == 9 )
      {
        v21 = 1;
      }
      else
      {
        if ( v12 && v12 != v22 )
          goto LABEL_132;
        v12 = (_QWORD *)*v19;
      }
    }
    v19 += 3;
  }
  while ( v20 != v19 );
  if ( !v12 )
  {
LABEL_158:
    v12 = (_QWORD *)sub_1599EF0(*(_QWORD *)a1);
    v7 = a4 & (v12 == 0);
LABEL_8:
    if ( !v7 )
      goto LABEL_120;
LABEL_9:
    v13 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    if ( v13 == 16 )
      v13 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
    if ( v13 != 11 )
      return 0;
    sub_14C2530((unsigned int)&v73, a1, v70.m128i_i32[0], 0, v71.m128i_i32[2], a1, v71.m128i_i64[0], a3);
    v65 = v74;
    if ( (unsigned int)v74 > 0x40 )
      v66 = sub_16A5940(&v73);
    else
      v66 = sub_39FAC40(v73);
    if ( v76 > 0x40 )
    {
      v67 = v66 + sub_16A5940(&v75);
      v12 = 0;
      if ( v67 != v65 )
        goto LABEL_124;
    }
    else if ( (unsigned int)sub_39FAC40(v75) + v66 != v65 )
    {
      v12 = 0;
LABEL_117:
      if ( v65 > 0x40 && v73 )
        j_j___libc_free_0_0(v73);
      goto LABEL_120;
    }
    v12 = (_QWORD *)sub_15A1070(*(_QWORD *)a1, &v75);
    if ( v76 <= 0x40 )
    {
LABEL_126:
      v65 = v74;
      goto LABEL_117;
    }
LABEL_124:
    if ( v75 )
      j_j___libc_free_0_0(v75);
    goto LABEL_126;
  }
  if ( v21 && !sub_13CB700((__int64)v12, a1, v71.m128i_i64[0]) )
  {
LABEL_132:
    if ( !a4 )
      return 0;
    goto LABEL_9;
  }
LABEL_120:
  if ( v12 == (_QWORD *)a1 )
    return sub_1599EF0(*v12);
  return (__int64)v12;
}
