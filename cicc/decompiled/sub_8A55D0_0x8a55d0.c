// Function: sub_8A55D0
// Address: 0x8a55d0
//
__m128i *__fastcall sub_8A55D0(
        __int64 a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        int a8,
        int *a9,
        __m128i *a10)
{
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // rbx
  __m128i *v13; // r15
  unsigned __int8 v14; // r12
  bool v15; // r12
  char v16; // al
  __m128i *v17; // rax
  __m128i **v18; // rsi
  unsigned __int8 v19; // di
  char v20; // al
  char v21; // al
  char v22; // al
  __m128i *v23; // rdx
  __m128i *v24; // r12
  int *v25; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  char i; // al
  __int64 v30; // rdi
  char v31; // al
  int v32; // eax
  __int64 v33; // rdi
  __m128i *v34; // r12
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // eax
  int v40; // eax
  char v41; // al
  __int64 v42; // rax
  __m128i *v43; // rax
  unsigned __int8 v44; // r14
  __int64 v45; // r15
  __int64 *v46; // r13
  __int64 v47; // rsi
  _BYTE *v48; // rbx
  unsigned __int8 v49; // di
  char v50; // al
  __m128i *v51; // rax
  bool v52; // cl
  __m128i *v53; // rax
  __int64 v54; // [rsp+8h] [rbp-E8h]
  __m128i *v55; // [rsp+10h] [rbp-E0h]
  const __m128i *v56; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+30h] [rbp-C0h]
  __int64 v59; // [rsp+38h] [rbp-B8h]
  __int64 v60; // [rsp+38h] [rbp-B8h]
  __int64 v61; // [rsp+40h] [rbp-B0h]
  __int64 v62; // [rsp+48h] [rbp-A8h]
  _BOOL4 v63; // [rsp+50h] [rbp-A0h]
  unsigned int v64; // [rsp+58h] [rbp-98h]
  _BOOL4 v65; // [rsp+5Ch] [rbp-94h]
  __int64 v66; // [rsp+60h] [rbp-90h]
  bool v67; // [rsp+6Bh] [rbp-85h]
  unsigned int v68; // [rsp+6Ch] [rbp-84h]
  __int64 v69; // [rsp+70h] [rbp-80h]
  __m128i *v70; // [rsp+78h] [rbp-78h]
  _BYTE v72[12]; // [rsp+94h] [rbp-5Ch]
  bool v73; // [rsp+94h] [rbp-5Ch]
  unsigned int v74; // [rsp+A0h] [rbp-50h] BYREF
  int v75; // [rsp+A4h] [rbp-4Ch] BYREF
  __m128i *v76; // [rsp+A8h] [rbp-48h] BYREF
  unsigned __int64 v77; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v78[14]; // [rsp+B8h] [rbp-38h] BYREF

  v10 = a3;
  v67 = a3 != 0;
  v66 = a4;
  v65 = (a8 & 0x40) != 0;
  v70 = (__m128i *)a5;
  v69 = a6;
  v68 = a8 & 2;
  if ( a1 )
  {
    a3 = *(unsigned __int8 *)(a1 + 80);
    switch ( (char)a3 )
    {
      case 4:
      case 5:
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
        goto LABEL_59;
      case 6:
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
        goto LABEL_59;
      case 9:
      case 10:
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
        goto LABEL_59;
      case 19:
      case 20:
      case 21:
      case 22:
        v27 = *(_QWORD *)(a1 + 88);
LABEL_59:
        if ( !v27 )
          goto LABEL_3;
        v63 = (*(_BYTE *)(v27 + 160) & 0x10) != 0;
        break;
      default:
LABEL_3:
        v63 = 0;
        break;
    }
    a6 = a8 & 2;
    if ( (a8 & 2) != 0 )
      v68 = (_BYTE)a3 != 21;
  }
  else
  {
    a5 = a8 & 2;
    v63 = 0;
    v68 = v68 != 0;
  }
  v11 = (__int64)a2;
  v12 = v10;
  v13 = 0;
  v56 = 0;
  *(_DWORD *)&v72[8] = 0;
  *(_QWORD *)v72 = a8 & 0xFFFFFFFC;
  v64 = 0;
  while ( 2 )
  {
    v77 = 0;
    v58 = 0;
    if ( v11 )
      v58 = *(_QWORD *)v11;
    if ( !v12 || (*(_BYTE *)(v12 + 56) & 0x10) == 0 )
    {
      if ( (a8 & 0x40) != 0 )
      {
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( qword_4F077A0 )
            goto LABEL_121;
        }
        else if ( HIDWORD(qword_4F077B4) && qword_4F077A8 <= 0x1116Fu )
        {
          goto LABEL_121;
        }
      }
      if ( v66 )
      {
        v39 = 1;
        if ( (*(_BYTE *)(v66 + 56) & 0x10) == 0 )
          v39 = v63;
        v63 = v39;
      }
LABEL_121:
      if ( (a8 & 0x400) == 0 )
        goto LABEL_16;
      goto LABEL_13;
    }
    a4 = a8 & 0x400;
    if ( (a8 & 0x400) == 0 )
    {
LABEL_46:
      v63 = 1;
      goto LABEL_47;
    }
    v63 = 1;
LABEL_13:
    if ( v11 )
    {
      if ( *(_BYTE *)(v11 + 8) != 3 && (*(_BYTE *)(v11 + 24) & 0x10) == 0 )
      {
LABEL_16:
        if ( !v12 && v67 && !v63 && v11 )
        {
          if ( *(_BYTE *)(v11 + 8) != 3 )
            goto LABEL_57;
          v14 = 0;
          goto LABEL_22;
        }
        goto LABEL_47;
      }
      goto LABEL_46;
    }
LABEL_47:
    v14 = v12 != 0;
    if ( (!v65 & v64) != 0 )
    {
      if ( v11 )
      {
        if ( (v14 & (*(_BYTE *)(v11 + 8) == 3)) != 0 )
        {
          if ( (*(_BYTE *)(v12 + 56) & 0x10) != 0 && !*(_QWORD *)v11 )
          {
            if ( !v10 )
              return v13;
LABEL_53:
            if ( !*(_QWORD *)v12 )
              return v13;
LABEL_54:
            if ( (a8 & 0x100) != 0 && (!a1 || *(_BYTE *)(a1 + 80) == 20) )
              return v13;
LABEL_57:
            *a9 = 1;
            return v13;
          }
          v14 &= *(_BYTE *)(v11 + 8) == 3;
        }
        goto LABEL_22;
      }
    }
    else if ( v11 )
    {
LABEL_22:
      if ( !*(_QWORD *)(v11 + 16) )
        goto LABEL_25;
      v78[0] = 0;
      a10[5].m128i_i32[3] = 0;
      if ( (a8 & 0x2000) != 0 )
        goto LABEL_24;
      v40 = sub_869530(*(_QWORD *)(v11 + 16), v69, v70, (__int64 *)&v77, a8 & 0xFFFC, (__int64)a10, (int *)v78);
      if ( v40 )
      {
        if ( v78[0] )
          *a9 = 1;
LABEL_24:
        v56 = (const __m128i *)v11;
        goto LABEL_25;
      }
      a6 = a10[5].m128i_u32[3];
      if ( (_DWORD)a6 && (a8 & 0x4000) != 0 )
      {
        a5 = v78[0];
        v65 = 1;
        if ( v78[0] )
          *a9 = 1;
      }
      else if ( v78[0] )
      {
        *a9 = 1;
      }
      if ( !v67 || !v14 )
      {
        v56 = (const __m128i *)v11;
        v34 = *(__m128i **)&v72[4];
        goto LABEL_102;
      }
      if ( !a1 || (*(_BYTE *)(v12 + 56) & 1) == 0 )
      {
        v56 = (const __m128i *)v11;
        v34 = *(__m128i **)&v72[4];
        goto LABEL_197;
      }
      v56 = (const __m128i *)v11;
LABEL_166:
      a3 = *(unsigned __int8 *)(a1 + 80);
      a4 = (unsigned int)(a3 - 21);
      if ( (unsigned __int8)(a3 - 21) > 1u && (_BYTE)a3 != 19 || (a3 = a8 & 0x140, (_DWORD)a3 == 320) && v56 )
      {
        if ( v40 )
          goto LABEL_25;
LABEL_111:
        v34 = *(__m128i **)&v72[4];
      }
      else
      {
        v49 = 0;
        v50 = *(_BYTE *)(*(_QWORD *)(v12 + 8) + 80LL);
        if ( v50 != 3 )
          v49 = (v50 != 2) + 1;
        v34 = (__m128i *)sub_725090(v49);
        sub_8AEEA0(a1, v34, v12, v10);
        if ( v13 )
        {
          if ( (*(_BYTE *)(v12 + 56) & 2) != 0 )
          {
            sub_8A4F30((__int64)v34, v12, v13, v10, v13, v10, a7, *(unsigned int *)v72, a9, a10);
            a4 = (unsigned int)*a9;
            if ( (_DWORD)a4 )
              return v13;
          }
          **(_QWORD **)&v72[4] = v34;
        }
        else
        {
          v13 = v34;
        }
        if ( (*(_BYTE *)(v12 + 56) & 0x10) == 0 )
        {
          if ( v34->m128i_i8[8] != 3 )
          {
            v12 = *(_QWORD *)v12;
            if ( v66 )
              v66 = *(_QWORD *)v66;
          }
          goto LABEL_102;
        }
        v11 = (__int64)v34;
      }
LABEL_197:
      if ( v11 )
        goto LABEL_102;
      *(_QWORD *)&v72[4] = v34;
      v52 = v12 != 0 && v67;
      if ( *a9 )
      {
LABEL_199:
        if ( !v52 )
          return v13;
LABEL_189:
        if ( (*(_BYTE *)(v12 + 56) & 0x10) == 0 )
          goto LABEL_54;
        goto LABEL_53;
      }
LABEL_205:
      if ( v56 && v65 )
      {
        v73 = v52;
        v53 = (__m128i *)sub_725090(v56->m128i_u8[8]);
        *v53 = _mm_loadu_si128(v56);
        v53[1] = _mm_loadu_si128(v56 + 1);
        v53[2] = _mm_loadu_si128(v56 + 2);
        v53[3].m128i_i64[0] = v56[3].m128i_i64[0];
        if ( v73 && (*(_BYTE *)(v12 + 56) & 0x10) != 0 )
          v53[1].m128i_i8[8] |= 8u;
        v53->m128i_i64[0] = 0;
        if ( !v13 )
          return v53;
        **(_QWORD **)&v72[4] = v53;
        return v13;
      }
      goto LABEL_199;
    }
    if ( v67 && v12 && a1 && (*(_BYTE *)(v12 + 56) & 1) != 0 )
    {
      v40 = 1;
      goto LABEL_166;
    }
    while ( 1 )
    {
LABEL_25:
      v15 = v67 && v12 != 0;
      if ( !v15 )
      {
        if ( !v11 )
        {
          v52 = 0;
          if ( *a9 )
            return v13;
          goto LABEL_205;
        }
        v19 = *(_BYTE *)(v11 + 8);
        if ( !v10 )
        {
          v76 = (__m128i *)sub_725090(v19);
          a3 = (__int64)v76;
          goto LABEL_35;
        }
        if ( v19 != 3 )
        {
          if ( !v12 )
          {
            a6 = v65;
            if ( v65 )
              goto LABEL_100;
            a5 = a8 & 0x2000;
            *a9 = 1;
            if ( (a8 & 0x2000) != 0 )
              return v13;
            goto LABEL_110;
          }
          goto LABEL_34;
        }
        if ( !v65 )
        {
          if ( !v12 )
            goto LABEL_100;
          goto LABEL_97;
        }
        goto LABEL_124;
      }
      v16 = *(_BYTE *)(v12 + 56);
      if ( (v16 & 0x10) == 0 )
        goto LABEL_32;
      a3 = v64;
      if ( v64 )
        goto LABEL_32;
      if ( !v11 )
      {
        v17 = (__m128i *)sub_725090(3u);
        v76 = v17;
        if ( v13 )
        {
LABEL_31:
          v64 = 1;
          v18 = *(__m128i ***)&v72[4];
          *(_QWORD *)&v72[4] = v17;
          *v18 = v17;
LABEL_32:
          if ( v11 )
            goto LABEL_33;
          v52 = v67 && v12 != 0;
          v17 = v13;
        }
        else
        {
          *(_QWORD *)&v72[4] = v17;
          v52 = v67 && v12 != 0;
        }
        v13 = v17;
        if ( *a9 )
          goto LABEL_189;
        goto LABEL_205;
      }
      if ( *(_BYTE *)(v11 + 8) == 3 )
      {
        a3 = v65;
        if ( !v65 )
          goto LABEL_98;
LABEL_124:
        v34 = *(__m128i **)&v72[4];
        if ( !v12 )
          goto LABEL_101;
        goto LABEL_125;
      }
      v17 = (__m128i *)sub_725090(3u);
      v76 = v17;
      if ( v13 )
        goto LABEL_31;
      *(_QWORD *)&v72[4] = v17;
      v13 = v17;
      v64 = 1;
LABEL_33:
      v19 = *(_BYTE *)(v11 + 8);
      if ( v19 == 3 )
      {
        if ( !v65 )
        {
LABEL_97:
          v16 = *(_BYTE *)(v12 + 56);
LABEL_98:
          if ( (v16 & 0x10) == 0 || v64 )
          {
LABEL_100:
            v34 = *(__m128i **)&v72[4];
            goto LABEL_101;
          }
        }
LABEL_125:
        v19 = 3;
      }
LABEL_34:
      v76 = (__m128i *)sub_725090(v19);
      a3 = (__int64)v76;
      v15 = (*(_BYTE *)(v12 + 56) & 0x10) != 0;
LABEL_35:
      v20 = (8 * v15) | *(_BYTE *)(a3 + 24) & 0xF7;
      *(_BYTE *)(a3 + 24) = v20;
      v21 = *(_BYTE *)(v11 + 24) & 0x10 | v20 & 0xEF;
      *(_BYTE *)(a3 + 24) = v21;
      v22 = *(_BYTE *)(v11 + 24) & 2 | v21 & 0xFD;
      *(_BYTE *)(a3 + 24) = v22;
      a4 = *(unsigned __int8 *)(v11 + 8);
      if ( (_BYTE)a4 == 2 )
        goto LABEL_112;
      if ( (unsigned __int8)a4 <= 2u )
      {
        if ( (_BYTE)a4 )
        {
          a4 = *(_QWORD *)(v11 + 32);
          *(_QWORD *)(a3 + 32) = a4;
          if ( (*(_BYTE *)(v11 + 24) & 0x40) != 0 )
            *(_BYTE *)(a3 + 24) = v22 | 0x40;
          goto LABEL_40;
        }
LABEL_112:
        *(_QWORD *)(a3 + 32) = *(_QWORD *)(v11 + 32);
        goto LABEL_40;
      }
      if ( (_BYTE)a4 != 3 )
        sub_721090();
      v64 = 1;
      *(_BYTE *)(a3 + 24) = v22 & 0xF7;
      while ( 1 )
      {
LABEL_40:
        if ( *(_BYTE *)(v11 + 8) == 3 )
          goto LABEL_75;
        v23 = a2;
        v24 = v76;
        if ( (a8 & 0x408) == 0 )
          v23 = v13;
        sub_8A4F30((__int64)v76, v12, v23, v10, v70, v69, a7, *(unsigned int *)v72, a9, a10);
        v25 = a9;
        if ( (v24[1].m128i_i8[8] & 0x40) != 0 )
          break;
LABEL_72:
        a3 = v68;
        v32 = *v25;
        if ( !v68 )
          goto LABEL_76;
        if ( v32 )
          return v13;
        sub_690F40((__int64)v76, v11);
LABEL_75:
        v32 = *a9;
LABEL_76:
        if ( v32 )
          return v13;
        v33 = *(_QWORD *)(v11 + 16);
        v34 = v76;
        if ( v33 && (v76[1].m128i_i8[8] & 0x10) != 0 && (a8 & 0x82000) != 0 )
        {
          if ( a10[2].m128i_i64[0] || a10[3].m128i_i64[1] )
            v34[1].m128i_i64[0] = (__int64)sub_892C00(v33, a10->m128i_i64);
          else
            v76[1].m128i_i64[0] = v33;
        }
        if ( v13 )
          **(_QWORD **)&v72[4] = v34;
        else
          v13 = v34;
        if ( v10 )
        {
          if ( (*(_BYTE *)(v12 + 56) & 0x10) == 0 && *(_BYTE *)(v11 + 8) != 3 )
          {
            v12 = *(_QWORD *)v12;
            if ( v66 )
              v66 = *(_QWORD *)v66;
          }
        }
        if ( !v34->m128i_i64[0] )
          goto LABEL_101;
        v76 = (__m128i *)v34->m128i_i64[0];
        *(_QWORD *)&v72[4] = v34;
      }
      if ( *a9 )
        return v13;
      v28 = v24[2].m128i_i64[0];
      v74 = 0;
      *(_QWORD *)v78 = v28;
      a4 = *(_QWORD *)(v28 + 128);
      for ( i = *(_BYTE *)(a4 + 140); i == 12; i = *(_BYTE *)(a4 + 140) )
        a4 = *(_QWORD *)(a4 + 160);
      if ( i == 14 )
        goto LABEL_71;
      if ( i != 2 )
        goto LABEL_70;
      v59 = a4;
      while ( (unsigned int)sub_72E9D0((_BYTE *)v28, v78, &v75) )
        v28 = *(_QWORD *)v78;
      v30 = *(_QWORD *)v78;
      a4 = v59;
      v31 = *(_BYTE *)(*(_QWORD *)v78 + 173LL);
      if ( v31 == 12 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v78 + 176LL) != 12 )
          goto LABEL_71;
        v30 = *(_QWORD *)(*(_QWORD *)v78 + 184LL);
        *(_QWORD *)v78 = v30;
        v41 = *(_BYTE *)(v30 + 173);
        if ( v41 == 12 )
          goto LABEL_71;
        if ( v41 != 1 )
          goto LABEL_70;
      }
      else if ( v31 != 1 )
      {
        goto LABEL_70;
      }
      v42 = sub_620FA0(v30, &v74);
      a4 = v74;
      v62 = v42;
      if ( v74 || (a4 = v59, v42 < 0) )
      {
LABEL_70:
        *a9 = 1;
LABEL_71:
        v25 = a9;
        goto LABEL_72;
      }
      if ( v42 )
      {
        v43 = v24;
        v61 = v11;
        v55 = v13;
        v44 = *(_BYTE *)(v59 + 160);
        v45 = 0;
        v54 = v10;
        v46 = (__int64 *)&v76;
        v60 = v12;
        while ( 1 )
        {
          if ( v43 )
          {
            v43[1].m128i_i8[8] &= ~0x40u;
            v48 = *(_BYTE **)(*v46 + 32);
            if ( !v48 )
              v48 = sub_724D80(1);
          }
          else
          {
            v51 = (__m128i *)sub_725090(1u);
            *v46 = (__int64)v51;
            *v51 = _mm_loadu_si128(v24);
            v51[1] = _mm_loadu_si128(v24 + 1);
            v51[2] = _mm_loadu_si128(v24 + 2);
            v51[3].m128i_i64[0] = v24[3].m128i_i64[0];
            *(_QWORD *)*v46 = 0;
            v48 = sub_724D80(1);
          }
          v47 = v45++;
          sub_72BAF0((__int64)v48, v47, v44);
          a3 = *v46;
          *(_QWORD *)(*v46 + 32) = v48;
          v46 = (__int64 *)*v46;
          if ( v62 <= v45 )
            break;
          v43 = (__m128i *)*v46;
        }
        v11 = v61;
        v12 = v60;
        v13 = v55;
        v10 = v54;
        if ( !v76 )
          goto LABEL_100;
        goto LABEL_71;
      }
      sub_725130(v24->m128i_i64);
      v76 = 0;
      v34 = *(__m128i **)&v72[4];
LABEL_101:
      if ( (a8 & 0x2000) != 0 )
        break;
      *(_QWORD *)&v72[4] = v34;
LABEL_110:
      sub_867630(v77, 0, a3, a4, a5, a6);
      if ( !(unsigned int)sub_866C00(v77, 0, v35, v36, v37, v38) )
        goto LABEL_111;
    }
LABEL_102:
    if ( !*a9 )
    {
      *(_QWORD *)&v72[4] = v34;
      v11 = v58;
      continue;
    }
    return v13;
  }
}
