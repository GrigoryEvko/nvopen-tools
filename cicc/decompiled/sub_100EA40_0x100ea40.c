// Function: sub_100EA40
// Address: 0x100ea40
//
unsigned __int8 *__fastcall sub_100EA40(
        unsigned __int8 *a1,
        unsigned __int8 **a2,
        unsigned __int64 a3,
        const __m128i *a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int8 **v6; // r12
  __int64 v8; // rax
  unsigned __int8 *v9; // r14
  __int64 v11; // r15
  __int64 v12; // r8
  __int64 v13; // rcx
  unsigned __int8 **v14; // rax
  unsigned __int8 **v15; // rcx
  __int64 *v16; // rsi
  __int64 **v17; // rax
  __int64 **v18; // rdx
  __int64 **i; // rdx
  unsigned __int64 v20; // rax
  unsigned __int8 *v21; // r12
  bool v22; // zf
  __int64 v23; // rax
  _BOOL8 v24; // rdx
  _BOOL4 v25; // r14d
  bool v26; // al
  _BOOL8 v27; // rdx
  _BOOL4 v28; // r14d
  bool v29; // al
  unsigned __int8 v30; // al
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  unsigned __int8 **v34; // rbx
  char v35; // al
  unsigned __int8 *v36; // r15
  unsigned int v37; // eax
  char v38; // al
  _BOOL8 v39; // rcx
  __int64 v40; // r13
  __int64 *v41; // r12
  _BOOL8 v42; // rcx
  __int64 v43; // r13
  __int64 *v44; // r12
  _BOOL8 v45; // rdx
  _BOOL4 v46; // r14d
  bool v47; // al
  char v48; // al
  char v49; // al
  _BOOL4 v50; // r14d
  unsigned __int8 *v51; // r13
  __int64 *v52; // r12
  _BOOL8 v53; // rcx
  char v54; // r13
  __int64 **v55; // rax
  _BOOL8 v56; // rdx
  char v57; // al
  __int64 v58; // rdx
  char v59; // [rsp+0h] [rbp-F0h]
  int v60; // [rsp+8h] [rbp-E8h]
  int v61; // [rsp+8h] [rbp-E8h]
  char v62; // [rsp+8h] [rbp-E8h]
  unsigned int v63; // [rsp+8h] [rbp-E8h]
  int v64; // [rsp+8h] [rbp-E8h]
  unsigned int v65; // [rsp+8h] [rbp-E8h]
  __int64 *v66; // [rsp+18h] [rbp-D8h] BYREF
  __m128i v67; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v68; // [rsp+30h] [rbp-C0h]
  __m128i v69; // [rsp+40h] [rbp-B0h]
  __m128i v70; // [rsp+50h] [rbp-A0h]
  __int64 v71; // [rsp+60h] [rbp-90h]
  __int64 **v72; // [rsp+70h] [rbp-80h] BYREF
  __int64 v73; // [rsp+78h] [rbp-78h]
  _BYTE v74[112]; // [rsp+80h] [rbp-70h] BYREF

  v6 = a2;
  if ( a4[2].m128i_i64[1] )
  {
    v71 = a4[4].m128i_i64[0];
    v67 = _mm_loadu_si128(a4);
    v68 = _mm_loadu_si128(a4 + 1);
    v69 = _mm_loadu_si128(a4 + 2);
  }
  else
  {
    v8 = a4[4].m128i_i64[0];
    v67 = _mm_loadu_si128(a4);
    v69.m128i_i64[0] = _mm_loadu_si128(a4 + 2).m128i_u64[0];
    v71 = v8;
    v69.m128i_i64[1] = (__int64)a1;
    v68 = _mm_loadu_si128(a4 + 1);
  }
  v70 = _mm_loadu_si128(a4 + 3);
  switch ( *a1 )
  {
    case ')':
      v21 = *a2;
      if ( **a2 > 0x15u || (v9 = (unsigned __int8 *)sub_96E680(12, (__int64)v21)) == 0 )
      {
        v72 = &v66;
        v22 = (unsigned __int8)sub_995E90(&v72, (unsigned __int64)v21, a3, (__int64)a4, a5) == 0;
        v23 = 0;
        if ( !v22 )
          return (unsigned __int8 *)v66;
        return (unsigned __int8 *)v23;
      }
      return v9;
    case '*':
      v27 = 0;
      v28 = 0;
      if ( (_BYTE)v71 )
      {
        v61 = a5;
        v28 = sub_B448F0((__int64)a1);
        v29 = sub_B44900((__int64)a1);
        LODWORD(a5) = v61;
        v27 = v29;
      }
      return (unsigned __int8 *)sub_101B9B0(*a2, a2[1], v27, v28, &v67, (unsigned int)a5);
    case '+':
      v38 = sub_B45210((__int64)a1);
      return sub_100E540((__int64 *)*a2, a2[1], v38, &v67, 0, 1);
    case ',':
      v24 = 0;
      v25 = 0;
      if ( (_BYTE)v71 )
      {
        v60 = a5;
        v25 = sub_B448F0((__int64)a1);
        v26 = sub_B44900((__int64)a1);
        LODWORD(a5) = v60;
        v24 = v26;
      }
      return (unsigned __int8 *)sub_101BE30(*a2, a2[1], v24, v25, &v67, (unsigned int)a5);
    case '-':
      v57 = sub_B45210((__int64)a1);
      return (unsigned __int8 *)sub_10088F0((__int64 *)*a2, (__int64 *)a2[1], v57, &v67, 0, 1);
    case '.':
      v56 = 0;
      if ( (_BYTE)v71 )
        v56 = sub_B44900((__int64)a1);
      return (unsigned __int8 *)sub_101E3C0(*a2, a2[1], v56, &v67);
    case '/':
      v54 = sub_B45210((__int64)a1);
      v55 = (__int64 **)a2[1];
      v66 = (__int64 *)*a2;
      v72 = v55;
      v9 = (unsigned __int8 *)sub_FFE3E0(0x12u, (_BYTE **)&v66, (_BYTE **)&v72, v67.m128i_i64);
      if ( !v9 )
        return sub_1009850((__int64)v66, (__int64)v72, v54, &v67, 0, 1);
      return v9;
    case '0':
      v53 = 0;
      if ( (_BYTE)v71 )
        v53 = (a1[1] & 2) != 0;
      return (unsigned __int8 *)sub_101A620(19, *a2, a2[1], v53, &v67, (unsigned int)a5);
    case '1':
      v50 = 0;
      if ( (_BYTE)v71 )
        v50 = (a1[1] & 2) != 0;
      v51 = a2[1];
      v52 = (__int64 *)*a2;
      v65 = a5;
      if ( sub_98F660(*a2, v51, 1, 1) )
        return (unsigned __int8 *)sub_AD62B0(v52[1]);
      else
        return (unsigned __int8 *)sub_101A620(20, v52, v51, v50, &v67, v65);
    case '2':
      v49 = sub_B45210((__int64)a1);
      return sub_1009F30(*a2, a2[1], v49, v67.m128i_i64, 0, 1);
    case '3':
      return (unsigned __int8 *)sub_101AA20(22, *a2, a2[1], &v67);
    case '4':
      return (unsigned __int8 *)sub_101AF30(*a2, a2[1], &v67, (unsigned int)a5);
    case '5':
      v48 = sub_B45210((__int64)a1);
      return sub_10091D0((__int64 *)*a2, a2[1], v48, v67.m128i_i64, 0, 1);
    case '6':
      v45 = 0;
      v46 = 0;
      if ( (_BYTE)v71 )
      {
        v64 = a5;
        v46 = sub_B448F0((__int64)a1);
        v47 = sub_B44900((__int64)a1);
        LODWORD(a5) = v64;
        v45 = v47;
      }
      return (unsigned __int8 *)sub_101D1E0(*a2, a2[1], v45, v46, &v67, (unsigned int)a5);
    case '7':
      v42 = 0;
      if ( (_BYTE)v71 )
        v42 = (a1[1] & 2) != 0;
      v43 = (__int64)a2[1];
      v44 = (__int64 *)*a2;
      v9 = (unsigned __int8 *)sub_101D570(26, *a2, v43, v42, &v67, (unsigned int)a5);
      if ( !v9 )
        return (unsigned __int8 *)sub_1006470((unsigned __int8 *)v44, v43, &v67);
      return v9;
    case '8':
      v39 = 0;
      if ( (_BYTE)v71 )
        v39 = (a1[1] & 2) != 0;
      v40 = (__int64)a2[1];
      v41 = (__int64 *)*a2;
      v9 = (unsigned __int8 *)sub_101D570(27, *a2, v40, v39, &v67, (unsigned int)a5);
      if ( !v9 )
        return (unsigned __int8 *)sub_1004A20((unsigned __int8 *)v41, v40, (__int64)&v67);
      return v9;
    case '9':
      return (unsigned __int8 *)sub_101D750(*a2, a2[1], &v67, (unsigned int)a5);
    case ':':
      return (unsigned __int8 *)sub_1010B00(*a2, a2[1], &v67, (unsigned int)a5);
    case ';':
      return (unsigned __int8 *)sub_101B6D0(*a2, a2[1], &v67, (unsigned int)a5);
    case '<':
      return 0;
    case '=':
      return (unsigned __int8 *)sub_1002AD0((__int64)a1, (__int64)*a2, v67.m128i_i64);
    case '?':
      v30 = sub_B4DE20((__int64)a1);
      return (unsigned __int8 *)sub_100C5A0(*((_QWORD *)a1 + 9), (__int64)*a2, a2 + 1, a3 - 1, v30, v67.m128i_i64);
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
      return (unsigned __int8 *)sub_1001480((unsigned int)*a1 - 29, *a2, *((_QWORD *)a1 + 1), v67.m128i_i64);
    case 'R':
      return (unsigned __int8 *)sub_1012FB0(
                                  *((_WORD *)a1 + 1) & 0x3F | ((unsigned __int64)((a1[1] & 2) != 0) << 32),
                                  *a2,
                                  a2[1],
                                  &v67);
    case 'S':
      v63 = a5;
      v37 = sub_B45210((__int64)a1);
      return (unsigned __int8 *)sub_1011B90(*((_WORD *)a1 + 1) & 0x3F, *a2, a2[1], v37, &v67, v63);
    case 'T':
      v34 = &a2[a3];
      if ( v34 == a2 )
        return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)a1 + 1));
      v59 = 0;
      v9 = 0;
      v62 = 0;
      do
      {
        v36 = *v6;
        if ( a1 != *v6 )
        {
          if ( *v36 == 13 )
          {
            v62 = 1;
          }
          else
          {
            v35 = sub_1003090((__int64)&v67, *v6);
            if ( v35 )
            {
              v59 = v35;
            }
            else
            {
              if ( v9 && v36 != v9 )
                return 0;
              v9 = v36;
            }
          }
        }
        ++v6;
      }
      while ( v34 != v6 );
      if ( v9 )
      {
        if ( (v62 || v59)
          && (!sub_FFE760((__int64)v9, (__int64)a1, v68.m128i_i64[1])
           || v59 && !sub_98ED70(v9, v69.m128i_i64[0], v69.m128i_i64[1], v68.m128i_i64[1], 0)) )
        {
          return 0;
        }
      }
      else if ( v59 )
      {
        return (unsigned __int8 *)sub_ACA8A0(*((__int64 ***)a1 + 1));
      }
      else
      {
        return (unsigned __int8 *)sub_ACADE0(*((__int64 ***)a1 + 1));
      }
      return v9;
    case 'U':
      if ( (a1[7] & 0x80u) == 0 )
        return (unsigned __int8 *)sub_10194C0((__int64)a1);
      v31 = sub_BD2BC0((__int64)a1);
      v33 = v31 + v32;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v33 >> 4) )
          goto LABEL_129;
      }
      else if ( (unsigned int)((v33 - sub_BD2BC0((__int64)a1)) >> 4) )
      {
        if ( (a1[7] & 0x80u) != 0 )
        {
          sub_BD2BC0((__int64)a1);
          if ( (a1[7] & 0x80u) == 0 )
            BUG();
          sub_BD2BC0((__int64)a1);
          return (unsigned __int8 *)sub_10194C0((__int64)a1);
        }
LABEL_129:
        BUG();
      }
      return (unsigned __int8 *)sub_10194C0((__int64)a1);
    case 'V':
      return (unsigned __int8 *)sub_101EAE0(*a2, a2[1], a2[2], &v67);
    case 'Z':
      return (unsigned __int8 *)sub_1004820((__int64)*a2, (__int64)a2[1], (__int64)&v67);
    case '[':
      return (unsigned __int8 *)sub_10031B0((__int64)*a2, (__int64)a2[1], (__int64)a2[2], (__int64)&v67);
    case '\\':
      return (unsigned __int8 *)sub_1003D50(
                                  (__int64)*a2,
                                  a2[1],
                                  *((_DWORD **)a1 + 9),
                                  *((unsigned int *)a1 + 20),
                                  *((__int64 ***)a1 + 1),
                                  (__int64)&v67,
                                  a5);
    case ']':
      return (unsigned __int8 *)sub_FFEEA0(*a2, *((unsigned int **)a1 + 9), *((unsigned int *)a1 + 20));
    case '^':
      return (unsigned __int8 *)sub_10046D0(
                                  (__int64)*a2,
                                  (__int64)a2[1],
                                  *((_DWORD **)a1 + 9),
                                  *((unsigned int *)a1 + 20),
                                  (__int64)&v67);
    case '`':
      return sub_1002A90(*a2, v67.m128i_i64);
    default:
      v11 = 8 * a3;
      v12 = (__int64)&a2[a3];
      v13 = (__int64)(8 * a3) >> 5;
      if ( v13 <= 0 )
      {
        v14 = a2;
      }
      else
      {
        v14 = a2;
        v15 = &a2[4 * v13];
        do
        {
          if ( **v14 > 0x15u )
            goto LABEL_14;
          if ( *v14[1] > 0x15u )
          {
            ++v14;
            goto LABEL_14;
          }
          if ( *v14[2] > 0x15u )
          {
            v14 += 2;
            goto LABEL_14;
          }
          if ( *v14[3] > 0x15u )
          {
            v14 += 3;
            goto LABEL_14;
          }
          v14 += 4;
        }
        while ( v15 != v14 );
      }
      v58 = v12 - (_QWORD)v14;
      if ( v12 - (_QWORD)v14 == 16 )
        goto LABEL_125;
      if ( v58 != 24 )
      {
        if ( v58 != 8 )
          goto LABEL_15;
LABEL_115:
        if ( **v14 <= 0x15u )
          goto LABEL_15;
        goto LABEL_14;
      }
      if ( **v14 <= 0x15u )
      {
        ++v14;
LABEL_125:
        if ( **v14 <= 0x15u )
        {
          ++v14;
          goto LABEL_115;
        }
      }
LABEL_14:
      v9 = 0;
      if ( (unsigned __int8 **)v12 != v14 )
        return v9;
LABEL_15:
      v72 = (__int64 **)v74;
      v16 = (__int64 *)v74;
      v73 = 0x800000000LL;
      if ( a3 )
      {
        v17 = (__int64 **)v74;
        v18 = (__int64 **)v74;
        if ( a3 > 8 )
        {
          sub_C8D5F0((__int64)&v72, v74, a3, 8u, v12, a6);
          v18 = v72;
          v12 = (__int64)&v6[a3];
          v17 = &v72[(unsigned int)v73];
        }
        for ( i = &v18[(unsigned __int64)v11 / 8]; i != v17; ++v17 )
        {
          if ( v17 )
            *v17 = 0;
        }
        LODWORD(v73) = a3;
        v16 = (__int64 *)v72;
      }
      if ( (unsigned __int8 **)v12 != v6 )
      {
        v20 = 0;
        do
        {
          v16[v20 / 8] = (__int64)v6[v20 / 8];
          v20 += 8LL;
        }
        while ( v20 != v11 );
        v16 = (__int64 *)v72;
      }
      v9 = (unsigned __int8 *)sub_97D230(a1, v16, (unsigned int)v73, v67.m128i_i64[0], (__int64 *)v67.m128i_i64[1], 1u);
      if ( v72 != (__int64 **)v74 )
        _libc_free(v72, v16);
      return v9;
  }
}
