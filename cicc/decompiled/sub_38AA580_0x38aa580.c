// Function: sub_38AA580
// Address: 0x38aa580
//
__int64 __fastcall sub_38AA580(
        __int64 a1,
        _BYTE **a2,
        unsigned __int64 a3,
        int a4,
        char a5,
        int a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        char a15,
        char a16,
        char a17,
        char a18)
{
  unsigned int v18; // r15d
  char v22; // r13
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // rdi
  unsigned int v28; // r15d
  const char *v30; // rax
  unsigned __int64 v31; // r14
  char v32; // cl
  int v33; // ebx
  char v34; // cl
  char v35; // r9
  char v36; // al
  char v37; // al
  unsigned int v38; // eax
  int v39; // eax
  double v40; // xmm4_8
  double v41; // xmm5_8
  int v42; // eax
  const void *v43; // rsi
  size_t v44; // rdx
  int *v45; // rax
  __m128i *v46; // rax
  char v47; // al
  char *v48; // rdi
  __int64 v49; // rax
  unsigned __int64 *v50; // rax
  unsigned __int64 *v51; // rsi
  _BYTE *v52; // rsi
  __int64 v53; // rcx
  unsigned __int64 *v54; // rax
  unsigned __int64 *v55; // rsi
  unsigned __int64 *v56; // rdx
  unsigned __int64 v57; // rdi
  __int64 v58; // r8
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // [rsp+8h] [rbp-128h]
  char v62; // [rsp+24h] [rbp-10Ch]
  char v64; // [rsp+3Bh] [rbp-F5h] BYREF
  unsigned int v65; // [rsp+3Ch] [rbp-F4h] BYREF
  __int64 v66; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v67; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v68; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v69; // [rsp+58h] [rbp-D8h] BYREF
  unsigned __int64 *v70[2]; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v71; // [rsp+70h] [rbp-C0h]
  char *v72; // [rsp+80h] [rbp-B0h] BYREF
  char *v73; // [rsp+88h] [rbp-A8h]
  __int64 v74; // [rsp+90h] [rbp-A0h] BYREF
  __m128i v75; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v76; // [rsp+B0h] [rbp-80h] BYREF
  __m128i *v77; // [rsp+C0h] [rbp-70h]
  __m128i *v78; // [rsp+C8h] [rbp-68h]
  __int64 v79; // [rsp+D0h] [rbp-60h]
  __int64 v80; // [rsp+D8h] [rbp-58h]
  __int64 v81; // [rsp+E0h] [rbp-50h]
  __int64 v82; // [rsp+E8h] [rbp-48h]
  __int64 v83; // [rsp+F0h] [rbp-40h]
  __int64 v84; // [rsp+F8h] [rbp-38h]

  v18 = a4 - 7;
  v62 = a6;
  if ( (unsigned int)(a4 - 7) <= 1 && a6 )
  {
    v75.m128i_i64[0] = (__int64)"symbol with local linkage must have default visibility";
    v76.m128i_i16[0] = 259;
    return (unsigned int)sub_38814C0(a1 + 8, a3, (__int64)&v75);
  }
  v66 = 0;
  v22 = sub_388BF60(a1, &v65);
  if ( v22 )
    return 1;
  if ( *(_DWORD *)(a1 + 64) == 42 )
  {
    v22 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  if ( (unsigned __int8)sub_388AD40(a1, &v64) )
    return 1;
  v23 = *(_QWORD *)(a1 + 56);
  v76.m128i_i16[0] = 259;
  v60 = v23;
  v75.m128i_i64[0] = (__int64)"expected type";
  if ( (unsigned __int8)sub_3891B00(a1, &v66, (__int64)&v75, 0) )
    return 1;
  v67 = 0;
  if ( !a5 || a4 != 9 && a4 )
  {
    if ( (unsigned __int8)sub_389C160((__int64 **)a1, v66, &v67) )
      return 1;
  }
  if ( *(_BYTE *)(v66 + 8) == 12 || !sub_1643F60(v66) )
  {
    v76.m128i_i8[1] = 1;
    v30 = "invalid type for global variable";
LABEL_25:
    v76.m128i_i8[0] = 3;
    v75.m128i_i64[0] = (__int64)v30;
    return (unsigned int)sub_38814C0(a1 + 8, v60, (__int64)&v75);
  }
  v24 = (__int64)a2[1];
  if ( !v24 )
  {
    v25 = *(_QWORD *)(a1 + 968);
    v26 = (__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3;
    if ( v25 )
    {
      v27 = a1 + 960;
      do
      {
        if ( (unsigned int)((__int64)(*(_QWORD *)(a1 + 1008) - *(_QWORD *)(a1 + 1000)) >> 3) > *(_DWORD *)(v25 + 32) )
        {
          v25 = *(_QWORD *)(v25 + 24);
        }
        else
        {
          v27 = v25;
          v25 = *(_QWORD *)(v25 + 16);
        }
      }
      while ( v25 );
      if ( a1 + 960 != v27 && (unsigned int)v26 >= *(_DWORD *)(v27 + 32) )
      {
        v31 = *(_QWORD *)(v27 + 40);
        v45 = sub_220F330((int *)v27, (_QWORD *)(a1 + 960));
        j_j___libc_free_0((unsigned __int64)v45);
        --*(_QWORD *)(a1 + 992);
        if ( v31 )
          goto LABEL_59;
      }
    }
LABEL_26:
    v76.m128i_i16[0] = 260;
    v75.m128i_i64[0] = (__int64)a2;
    v31 = (unsigned __int64)sub_1648A60(88, 1u);
    if ( v31 )
      sub_15E51E0(v31, *(_QWORD *)(a1 + 176), v66, 0, 0, 0, (__int64)&v75, 0, 0, v65, 0);
    goto LABEL_28;
  }
  v31 = sub_1632000(*(_QWORD *)(a1 + 176), (__int64)*a2, v24);
  if ( !v31 )
    goto LABEL_26;
  if ( sub_38942C0(a1 + 904, (__int64)a2) )
  {
LABEL_59:
    if ( v66 != *(_QWORD *)(v31 + 24) )
    {
      v76.m128i_i8[1] = 1;
      v30 = "forward reference and definition of global have different types";
      goto LABEL_25;
    }
    v53 = *(_QWORD *)(a1 + 176);
    v54 = *(unsigned __int64 **)(v31 + 64);
    v55 = (unsigned __int64 *)(v31 + 56);
    v56 = (unsigned __int64 *)(v53 + 8);
    if ( (unsigned __int64 *)(v53 + 8) != v54 && v56 != v55 && v56 != v54 && v55 != v54 )
    {
      v57 = *v54 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*(_QWORD *)(v31 + 56) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v54;
      *v54 = *v54 & 7 | *(_QWORD *)(v31 + 56) & 0xFFFFFFFFFFFFFFF8LL;
      v58 = *(_QWORD *)(v53 + 8);
      *(_QWORD *)(v57 + 8) = v56;
      v58 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v31 + 56) = v58 | *(_QWORD *)(v31 + 56) & 7LL;
      *(_QWORD *)(v58 + 8) = v55;
      *(_QWORD *)(v53 + 8) = v57 | *(_QWORD *)(v53 + 8) & 7LL;
    }
LABEL_28:
    if ( !a2[1] )
    {
      v75.m128i_i64[0] = v31;
      v52 = *(_BYTE **)(a1 + 1008);
      if ( v52 == *(_BYTE **)(a1 + 1016) )
      {
        sub_167C6C0(a1 + 1000, v52, &v75);
      }
      else
      {
        if ( v52 )
        {
          *(_QWORD *)v52 = v31;
          v52 = *(_BYTE **)(a1 + 1008);
        }
        *(_QWORD *)(a1 + 1008) = v52 + 8;
      }
    }
    if ( v67 )
      sub_15E5440(v31, v67);
    v32 = a4;
    v33 = a4 & 0xF;
    v34 = v32 & 0xF;
    v35 = v62 & 3;
    *(_BYTE *)(v31 + 80) = v64 & 1 | *(_BYTE *)(v31 + 80) & 0xFE;
    v36 = *(_BYTE *)(v31 + 32);
    if ( v18 > 1 )
    {
      v47 = v34 | v36 & 0xF0;
      *(_BYTE *)(v31 + 32) = v47;
      if ( (unsigned int)(v33 - 7) > 1 && ((v47 & 0x30) == 0 || v34 == 9) )
      {
        if ( !a16 )
        {
          *(_BYTE *)(v31 + 32) = *(_BYTE *)(v31 + 32) & 0xCF | (16 * v35);
LABEL_36:
          if ( v33 != 8 && ((*(_BYTE *)(v31 + 32) & 0x30) == 0 || v34 == 9) )
          {
LABEL_41:
            *(_BYTE *)(v31 + 80) = (2 * (v22 & 1)) | *(_BYTE *)(v31 + 80) & 0xFD;
            *(_WORD *)(v31 + 32) = *(_WORD *)(v31 + 32) & 0xE03F
                                 | ((a17 & 7) << 10)
                                 | ((a18 & 3) << 6)
                                 | ((a15 & 3) << 8);
            while ( *(_DWORD *)(a1 + 64) == 4 )
            {
              v39 = sub_3887100(a1 + 8);
              *(_DWORD *)(a1 + 64) = v39;
              switch ( v39 )
              {
                case 90:
                  v42 = sub_3887100(a1 + 8);
                  v43 = *(const void **)(a1 + 72);
                  v44 = *(_QWORD *)(a1 + 80);
                  *(_DWORD *)(a1 + 64) = v42;
                  sub_15E5D20(v31, v43, v44);
                  if ( (unsigned __int8)sub_388AF10(a1, 377, "expected global section string") )
                    return 1;
                  break;
                case 88:
                  v38 = sub_388C5A0(a1, (unsigned int *)&v75);
                  if ( (_BYTE)v38 )
                    return v38;
                  sub_15E4CC0(v31, v75.m128i_u32[0]);
                  break;
                case 376:
                  if ( (unsigned __int8)sub_38AA540(a1, v31, a7, a8, a9, a10, v40, v41, a13, a14) )
                    return 1;
                  break;
                default:
                  v38 = sub_3899C10(a1, *a2, (size_t)a2[1], (__int64 *)&v72);
                  if ( (_BYTE)v38 )
                    return v38;
                  if ( !v72 )
                  {
                    v59 = *(_QWORD *)(a1 + 56);
                    v76.m128i_i16[0] = 259;
                    v75.m128i_i64[0] = (__int64)"unknown global variable property!";
                    return (unsigned int)sub_38814C0(a1 + 8, v59, (__int64)&v75);
                  }
                  *(_QWORD *)(v31 + 48) = v72;
                  break;
              }
            }
            v77 = &v76;
            v76.m128i_i32[0] = 0;
            v75.m128i_i64[0] = 0;
            v76.m128i_i64[1] = 0;
            v78 = &v76;
            v79 = 0;
            v80 = 0;
            v81 = 0;
            v82 = 0;
            v83 = 0;
            v84 = 0;
            v68 = 0;
            v72 = 0;
            v73 = 0;
            v74 = 0;
            v28 = sub_388FCA0(a1, &v75, (__int64)&v72, 0, &v68);
            if ( !(_BYTE)v28 )
            {
              if ( !sub_1560CB0(&v75) )
              {
                v48 = v72;
                if ( v73 == v72 )
                  goto LABEL_89;
              }
              v49 = sub_1560BF0(*(__int64 **)a1, &v75);
              v69 = v31;
              *(_QWORD *)(v31 + 72) = v49;
              v50 = *(unsigned __int64 **)(a1 + 1144);
              v51 = (unsigned __int64 *)(a1 + 1136);
              if ( !v50 )
                goto LABEL_86;
              do
              {
                if ( v50[4] < v31 )
                {
                  v50 = (unsigned __int64 *)v50[3];
                }
                else
                {
                  v51 = v50;
                  v50 = (unsigned __int64 *)v50[2];
                }
              }
              while ( v50 );
              if ( v51 == (unsigned __int64 *)(a1 + 1136) || v51[4] > v31 )
              {
LABEL_86:
                v70[0] = &v69;
                v51 = sub_3898250((_QWORD *)(a1 + 1128), (__int64)v51, v70);
              }
              sub_3887600((__int64)(v51 + 5), &v72);
            }
            v48 = v72;
LABEL_89:
            if ( v48 )
              j_j___libc_free_0((unsigned __int64)v48);
            sub_3887AD0((_QWORD *)v76.m128i_i64[1]);
            return v28;
          }
          v37 = *(_BYTE *)(v31 + 33);
LABEL_40:
          *(_BYTE *)(v31 + 33) = v37 | 0x40;
          goto LABEL_41;
        }
        v37 = *(_BYTE *)(v31 + 33);
LABEL_34:
        v37 |= 0x40u;
        *(_BYTE *)(v31 + 33) = v37;
LABEL_35:
        *(_BYTE *)(v31 + 32) = (16 * v35) | *(_BYTE *)(v31 + 32) & 0xCF;
        if ( v33 == 7 )
          goto LABEL_40;
        goto LABEL_36;
      }
    }
    else
    {
      *(_BYTE *)(v31 + 32) = v34 | v36 & 0xC0;
    }
    v37 = *(_BYTE *)(v31 + 33) | 0x40;
    *(_BYTE *)(v31 + 33) = v37;
    if ( !a16 )
      goto LABEL_35;
    goto LABEL_34;
  }
  sub_8FD6D0((__int64)&v72, "redefinition of global '@", a2);
  if ( v73 == (char *)0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v46 = (__m128i *)sub_2241490((unsigned __int64 *)&v72, "'", 1u);
  v75.m128i_i64[0] = (__int64)&v76;
  if ( (__m128i *)v46->m128i_i64[0] == &v46[1] )
  {
    v76 = _mm_loadu_si128(v46 + 1);
  }
  else
  {
    v75.m128i_i64[0] = v46->m128i_i64[0];
    v76.m128i_i64[0] = v46[1].m128i_i64[0];
  }
  v75.m128i_i64[1] = v46->m128i_i64[1];
  v46->m128i_i64[0] = (__int64)v46[1].m128i_i64;
  v46->m128i_i64[1] = 0;
  v46[1].m128i_i8[0] = 0;
  v71 = 260;
  v70[0] = (unsigned __int64 *)&v75;
  v28 = sub_38814C0(a1 + 8, a3, (__int64)v70);
  if ( (__m128i *)v75.m128i_i64[0] != &v76 )
    j_j___libc_free_0(v75.m128i_u64[0]);
  if ( v72 != (char *)&v74 )
    j_j___libc_free_0((unsigned __int64)v72);
  return v28;
}
