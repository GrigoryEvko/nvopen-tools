// Function: sub_27CADD0
// Address: 0x27cadd0
//
char __fastcall sub_27CADD0(__int64 a1, __int64 *a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 *i; // rbx
  __int64 v10; // r15
  __m128i *v11; // rax
  char v12; // dl
  __int64 v13; // rdi
  bool v14; // zf
  _BYTE *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r15
  __int16 v18; // ax
  __int64 v19; // r15
  unsigned int v20; // r14d
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rsi
  __int64 v25; // r15
  __int64 *v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  const __m128i *v38; // r15
  __int64 *v39; // rax
  _QWORD *v40; // rax
  unsigned int v41; // eax
  unsigned __int64 v42; // rcx
  unsigned int v43; // edx
  unsigned int v44; // eax
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  _QWORD *v47; // rax
  unsigned int v48; // r15d
  __int64 v49; // r15
  unsigned int v50; // eax
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rbx
  const void *v56; // rsi
  char v57; // al
  __int64 *v59; // [rsp+0h] [rbp-A0h]
  _QWORD *v60; // [rsp+0h] [rbp-A0h]
  char v61; // [rsp+0h] [rbp-A0h]
  unsigned int v62; // [rsp+Ch] [rbp-94h]
  __int64 v63; // [rsp+10h] [rbp-90h]
  __int64 *v64; // [rsp+10h] [rbp-90h]
  __int64 v65; // [rsp+10h] [rbp-90h]
  __int64 *v66; // [rsp+18h] [rbp-88h]
  __int64 v67; // [rsp+20h] [rbp-80h]
  __int64 *v68; // [rsp+20h] [rbp-80h]
  __int64 v69; // [rsp+20h] [rbp-80h]
  _QWORD *v70; // [rsp+20h] [rbp-80h]
  _QWORD *v71; // [rsp+20h] [rbp-80h]
  __int64 v72; // [rsp+20h] [rbp-80h]
  __int64 v73; // [rsp+20h] [rbp-80h]
  unsigned int v74; // [rsp+20h] [rbp-80h]
  unsigned int v76; // [rsp+3Ch] [rbp-64h] BYREF
  unsigned __int64 v77; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v78; // [rsp+48h] [rbp-58h]
  unsigned __int64 v79; // [rsp+50h] [rbp-50h] BYREF
  __int64 v80; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v81; // [rsp+60h] [rbp-40h] BYREF
  __int64 *v82; // [rsp+68h] [rbp-38h]

  v7 = a4;
  for ( i = (__int64 *)a3; ; i = (__int64 *)(v17 + 32) )
  {
    v10 = *i;
    if ( !*(_BYTE *)(a5 + 28) )
      goto LABEL_8;
    v11 = *(__m128i **)(a5 + 8);
    a4 = *(unsigned int *)(a5 + 20);
    a3 = (__m128i *)((char *)v11 + 8 * a4);
    if ( v11 != a3 )
    {
      while ( v10 != v11->m128i_i64[0] )
      {
        v11 = (__m128i *)((char *)v11 + 8);
        if ( a3 == v11 )
          goto LABEL_25;
      }
      return (char)v11;
    }
LABEL_25:
    if ( (unsigned int)a4 < *(_DWORD *)(a5 + 16) )
    {
      *(_DWORD *)(a5 + 20) = a4 + 1;
      a3->m128i_i64[0] = v10;
      ++*(_QWORD *)a5;
    }
    else
    {
LABEL_8:
      LOBYTE(v11) = (unsigned __int8)sub_C8CC70(a5, *i, (__int64)a3, a4, a5, a6);
      if ( !v12 )
        return (char)v11;
    }
    if ( *(_BYTE *)v10 <= 0x1Cu )
      return (char)v11;
    v13 = *(_QWORD *)(v10 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v13 + 16);
    v14 = !sub_BCAC40(v13, 1);
    LOBYTE(v11) = *(_BYTE *)v10;
    if ( v14 )
      goto LABEL_28;
    if ( (_BYTE)v11 != 57 )
    {
      if ( (_BYTE)v11 != 86 )
        goto LABEL_28;
      v11 = *(__m128i **)(v10 - 96);
      if ( v11->m128i_i64[1] != *(_QWORD *)(v10 + 8) )
        return (char)v11;
      v15 = *(_BYTE **)(v10 - 32);
      if ( *v15 > 0x15u )
        return (char)v11;
      if ( !sub_AC30F0((__int64)v15) )
        break;
    }
    if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
      v16 = *(_QWORD *)(v10 - 8);
    else
      v16 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
    sub_27CADD0(a1, a2, v16, v7, a5);
    if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
      v17 = *(_QWORD *)(v10 - 8);
    else
      v17 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
  }
  LOBYTE(v11) = *(_BYTE *)v10;
LABEL_28:
  if ( (_BYTE)v11 == 82 )
  {
    v63 = *(_QWORD *)(v10 - 64);
    v11 = *(__m128i **)(v63 + 8);
    if ( v11->m128i_i8[8] == 12 )
    {
      v18 = *(_WORD *)(v10 + 2);
      v19 = *(_QWORD *)(v10 - 32);
      v20 = v18 & 0x3F;
      v67 = v19;
      v62 = v20;
      v21 = sub_DD8400((__int64)a2, v63);
      if ( sub_DADE90((__int64)a2, (__int64)v21, a1) )
      {
        v62 = sub_B52F50(v20);
        v22 = v63;
        v63 = v19;
        v67 = v22;
        goto LABEL_32;
      }
      v39 = sub_DD8400((__int64)a2, v19);
      LOBYTE(v11) = sub_DADE90((__int64)a2, (__int64)v39, a1);
      if ( (_BYTE)v11 )
      {
LABEL_32:
        v66 = sub_DD8400((__int64)a2, v63);
        if ( *((_WORD *)v66 + 12) == 8 )
        {
          switch ( v62 )
          {
            case '$':
            case '(':
              v29 = (__int64)sub_DD8400((__int64)a2, v67);
              break;
            case '%':
            case ')':
              v23 = sub_DA2C50((__int64)a2, *(_QWORD *)(v67 + 8), 1, 0);
              v59 = sub_DD8400((__int64)a2, v67);
              if ( !(unsigned __int8)sub_DDCBC0(a2, 13, v62 == 41, (__int64)v59, (__int64)v23, 0) )
                goto LABEL_35;
              v81 = (unsigned __int64 *)v59;
              v79 = (unsigned __int64)&v81;
              v82 = v23;
              v80 = 0x200000002LL;
              v40 = sub_DC7EB0(a2, (__int64)&v79, 0, 0);
              v29 = (__int64)v40;
              if ( (unsigned __int64 **)v79 != &v81 )
              {
                v70 = v40;
                _libc_free(v79);
                v29 = (__int64)v70;
              }
              break;
            case '&':
              if ( *(_BYTE *)v67 != 17 )
                goto LABEL_35;
              v41 = *(_DWORD *)(v67 + 32);
              LODWORD(v80) = v41;
              if ( v41 > 0x40 )
              {
                sub_C43780((__int64)&v79, (const void **)(v67 + 24));
                v41 = v80;
                if ( (unsigned int)v80 > 0x40 )
                {
                  sub_C43D10((__int64)&v79);
                  goto LABEL_60;
                }
              }
              else
              {
                v79 = *(_QWORD *)(v67 + 24);
              }
              v42 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v41;
              if ( !v41 )
                v42 = 0;
              v79 = v42 & ~v79;
LABEL_60:
              sub_C46250((__int64)&v79);
              v43 = v80;
              LODWORD(v80) = 0;
              v78 = v43;
              v77 = v79;
              if ( v43 > 0x40 )
              {
                v60 = (_QWORD *)v79;
                if ( v43 - (unsigned int)sub_C444A0((__int64)&v77) <= 0x40 && *v60 == 1 )
                {
                  j_j___libc_free_0_0((unsigned __int64)v60);
                  v57 = 1;
                  if ( (unsigned int)v80 <= 0x40 )
                    goto LABEL_62;
                }
                else
                {
                  if ( v60 )
                    j_j___libc_free_0_0((unsigned __int64)v60);
                  if ( (unsigned int)v80 <= 0x40 )
                    goto LABEL_35;
                  v57 = 0;
                }
                if ( v79 )
                {
                  v61 = v57;
                  j_j___libc_free_0_0(v79);
                  v57 = v61;
                }
                if ( !v57 )
                  goto LABEL_35;
              }
              else if ( v79 != 1 )
              {
                goto LABEL_35;
              }
LABEL_62:
              v44 = *(_DWORD *)(sub_D95540(*(_QWORD *)v66[4]) + 8) >> 8;
              LODWORD(v80) = v44;
              if ( v44 > 0x40 )
              {
                v74 = v44;
                goto LABEL_84;
              }
              v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
              if ( !v44 )
                v45 = 0;
              v79 = v45;
              v46 = ~(1LL << ((unsigned __int8)v44 - 1));
              goto LABEL_66;
            case '\'':
              if ( *(_BYTE *)v67 != 17 )
                goto LABEL_35;
              v48 = *(_DWORD *)(v67 + 32);
              if ( v48 > 0x40 )
              {
                if ( v48 - (unsigned int)sub_C444A0(v67 + 24) > 0x40 )
                  goto LABEL_35;
                v49 = **(_QWORD **)(v67 + 24);
              }
              else
              {
                v49 = *(_QWORD *)(v67 + 24);
              }
              if ( v49 )
                goto LABEL_35;
              v50 = *(_DWORD *)(sub_D95540(*(_QWORD *)v66[4]) + 8) >> 8;
              LODWORD(v80) = v50;
              if ( v50 <= 0x40 )
              {
                v51 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v50;
                if ( !v50 )
                  v51 = 0;
                v79 = v51;
                v46 = ~(1LL << ((unsigned __int8)v50 - 1));
LABEL_66:
                v79 &= v46;
                goto LABEL_67;
              }
              v74 = v50;
LABEL_84:
              sub_C43690((__int64)&v79, -1, 1);
              v46 = ~(1LL << ((unsigned __int8)v74 - 1));
              if ( (unsigned int)v80 <= 0x40 )
                goto LABEL_66;
              *(_QWORD *)(v79 + 8LL * ((v74 - 1) >> 6)) &= v46;
LABEL_67:
              v47 = sub_DA26C0(a2, (__int64)&v79);
              v29 = (__int64)v47;
              if ( (unsigned int)v80 > 0x40 && v79 )
              {
                v71 = v47;
                j_j___libc_free_0_0(v79);
                v29 = (__int64)v71;
              }
              break;
            default:
              goto LABEL_35;
          }
        }
        else
        {
LABEL_35:
          LOBYTE(v11) = v63;
          v77 = v63;
          v76 = v62;
          if ( *(_BYTE *)v63 != 44 )
            return (char)v11;
          v24 = *(_QWORD *)(v63 - 64);
          if ( !v24 )
            return (char)v11;
          v25 = *(_QWORD *)(v63 - 32);
          if ( !v25 )
            return (char)v11;
          v66 = sub_DD8400((__int64)a2, v24);
          v64 = sub_DD8400((__int64)a2, v25);
          v68 = sub_DD8400((__int64)a2, v67);
          LOBYTE(v11) = sub_DADE90((__int64)a2, (__int64)v66, a1);
          v26 = v68;
          if ( (_BYTE)v11 )
          {
            if ( *((_WORD *)v64 + 12) != 8 )
              return (char)v11;
            v80 = (__int64)&v76;
            v79 = (unsigned __int64)a2;
            v81 = &v77;
            v69 = sub_27CA700((__int64 *)&v79, 15, (__int64)v66, (__int64)v68);
            v27 = sub_B52F50(v76);
            v29 = v69;
            v76 = v27;
            v66 = v64;
          }
          else
          {
            v54 = (__int64)v64;
            v73 = (__int64)v64;
            v65 = (__int64)v26;
            LOBYTE(v11) = sub_DADE90((__int64)a2, v54, a1);
            if ( !(_BYTE)v11 )
              return (char)v11;
            LOBYTE(v11) = (_BYTE)v66;
            if ( *((_WORD *)v66 + 12) != 8 )
              return (char)v11;
            v80 = (__int64)&v76;
            v79 = (unsigned __int64)a2;
            v81 = &v77;
            v29 = sub_27CA700((__int64 *)&v79, 13, v73, v65);
          }
          LOBYTE(v11) = v76;
          v30 = v76 - 40;
          if ( (unsigned int)v30 > 1 )
            return (char)v11;
          if ( v76 == 41 )
          {
            if ( !v29 )
              return (char)v11;
            v72 = v29;
            v52 = sub_D95540(v29);
            v53 = sub_DA2C50((__int64)a2, v52, 1, 0);
            v11 = (__m128i *)sub_27CA700((__int64 *)&v79, 13, v72, (__int64)v53);
            v29 = (__int64)v11;
          }
          if ( !v29 )
            return (char)v11;
        }
        LOBYTE(v11) = (_BYTE)v66;
        if ( a1 == v66[6] && v66[5] == 2 )
        {
          v81 = (unsigned __int64 *)v29;
          v79 = *(_QWORD *)v66[4];
          v31 = sub_D33D80(v66, (__int64)a2, v30, v28, v29);
          v34 = *(unsigned int *)(v7 + 12);
          v82 = i;
          v80 = v31;
          v35 = *(unsigned int *)(v7 + 8);
          v36 = v35 + 1;
          if ( v35 + 1 > v34 )
          {
            v55 = *(_QWORD *)v7;
            v38 = (const __m128i *)&v79;
            v56 = (const void *)(v7 + 16);
            if ( *(_QWORD *)v7 > (unsigned __int64)&v79 || (unsigned __int64)&v79 >= v55 + 32 * v35 )
            {
              sub_C8D5F0(v7, v56, v36, 0x20u, v32, v33);
              v35 = *(unsigned int *)(v7 + 8);
              v37 = *(_QWORD *)v7;
            }
            else
            {
              sub_C8D5F0(v7, v56, v36, 0x20u, v32, v33);
              v37 = *(_QWORD *)v7;
              v35 = *(unsigned int *)(v7 + 8);
              v38 = (const __m128i *)((char *)&v79 + *(_QWORD *)v7 - v55);
            }
          }
          else
          {
            v37 = *(_QWORD *)v7;
            v38 = (const __m128i *)&v79;
          }
          v11 = (__m128i *)(v37 + 32 * v35);
          *v11 = _mm_loadu_si128(v38);
          v11[1] = _mm_loadu_si128(v38 + 1);
          ++*(_DWORD *)(v7 + 8);
        }
      }
    }
  }
  return (char)v11;
}
