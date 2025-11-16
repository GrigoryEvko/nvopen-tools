// Function: sub_635980
// Address: 0x635980
//
__int64 __fastcall sub_635980(__int64 **a1, __int64 *a2, __m128i *a3, _QWORD *a4, __int64 *a5)
{
  __int64 *v6; // r12
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  __int64 result; // rax
  __int64 m; // r12
  __int64 v12; // rdi
  __m128i *v13; // r14
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r13
  char v16; // al
  int v17; // eax
  __int64 v18; // r8
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rbx
  char k; // al
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  char v29; // al
  int v30; // edx
  int v31; // edx
  int v32; // eax
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rax
  char v35; // dl
  __int64 v36; // rax
  __int64 v37; // rdi
  _BOOL4 v38; // ebx
  __int8 v39; // bl
  __int64 v40; // rax
  __int8 v41; // al
  __int64 v42; // rbx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rdx
  char j; // al
  __int64 v51; // r9
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 *v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rdx
  __int64 v63; // rdi
  int v64; // eax
  __int64 n; // rax
  __int64 v66; // rax
  __int64 *v67; // r8
  __int64 v68; // rax
  __int64 v69; // r12
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rax
  bool v75; // [rsp+4h] [rbp-BCh]
  __int64 v76; // [rsp+8h] [rbp-B8h]
  __int64 v77; // [rsp+8h] [rbp-B8h]
  __int64 v78; // [rsp+8h] [rbp-B8h]
  __int64 *v79; // [rsp+10h] [rbp-B0h]
  __int64 v80; // [rsp+10h] [rbp-B0h]
  __int64 v81; // [rsp+10h] [rbp-B0h]
  __int64 v82; // [rsp+10h] [rbp-B0h]
  __int64 v83; // [rsp+10h] [rbp-B0h]
  __int64 *v84; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v86; // [rsp+28h] [rbp-98h]
  __int64 *i; // [rsp+30h] [rbp-90h]
  _BOOL4 v88; // [rsp+38h] [rbp-88h]
  char v89; // [rsp+3Eh] [rbp-82h]
  char v90; // [rsp+3Fh] [rbp-81h]
  unsigned __int64 v91; // [rsp+48h] [rbp-78h]
  int v94; // [rsp+68h] [rbp-58h]
  int v95; // [rsp+6Ch] [rbp-54h]
  __int64 *v96; // [rsp+70h] [rbp-50h] BYREF
  __int64 v97; // [rsp+78h] [rbp-48h] BYREF
  __int64 *v98; // [rsp+80h] [rbp-40h] BYREF
  __int64 v99[7]; // [rsp+88h] [rbp-38h] BYREF

  v6 = *a1;
  v7 = *a2;
  v84 = a2;
  v8 = *(_BYTE *)(*a2 + 140) == 12;
  v9 = *a2;
  v96 = v6;
  if ( v8 )
  {
    do
      v9 = *(_QWORD *)(v9 + 160);
    while ( *(_BYTE *)(v9 + 140) == 12 );
  }
  v97 = v9;
  if ( (unsigned int)sub_8D3880(v7) && (unsigned int)sub_6320D0((__int64)v6, a2, (__int64)a3, (__int64)a5) )
    goto LABEL_5;
  v90 = *((_BYTE *)v96 + 8);
  if ( !v90 )
  {
    a2 = (__int64 *)v97;
    if ( (unsigned int)sub_6964E0(v96, v97, a5) )
    {
LABEL_5:
      result = *v96;
      if ( *v96 )
      {
        if ( *(_BYTE *)(result + 8) == 3 )
          result = sub_6BBB10(v96);
      }
      goto LABEL_8;
    }
    v90 = *((_BYTE *)v96 + 8);
  }
  m = *(_QWORD *)(v97 + 160);
  if ( (a3[2].m128i_i8[8] & 0x40) != 0 )
  {
    *a5 = 0;
  }
  else
  {
    v57 = sub_724D50(10);
    v58 = v97;
    v59 = (__int64)v96;
    *a5 = v57;
    *(_QWORD *)(v57 + 128) = v58;
    v60 = (_QWORD *)sub_6E1A20(v59);
    v61 = (__int64)v96;
    *(_QWORD *)(*a5 + 64) = *v60;
    if ( *(_BYTE *)(v61 + 8) != 2 )
      *(_QWORD *)(*a5 + 112) = *(_QWORD *)sub_6E1A60(v61);
    v89 = v90 == 1;
    v62 = *a5;
    if ( (a3[2].m128i_i32[2] & 0x20008000) == 0x20000000 )
      *(_BYTE *)(v62 + 169) = (v89 << 6) | *(_BYTE *)(v62 + 169) & 0xBF;
    else
      *(_BYTE *)(v62 + 169) = (32 * v89) | *(_BYTE *)(v62 + 169) & 0xDF;
  }
  if ( v90 == 1 )
  {
    v56 = v96 + 5;
    a4 = v96 + 5;
    v96 = (__int64 *)v96[3];
    if ( !v96 && dword_4F077C4 != 2 && !dword_4F077C0 )
    {
      a2 = v56;
      sub_6851C0(29, v56);
    }
    v75 = (a3[2].m128i_i8[9] & 0x20) != 0;
  }
  else
  {
    v75 = 0;
    if ( (a3[2].m128i_i8[10] & 1) != 0 )
    {
      if ( (a3[2].m128i_i16[4] & 0x220) == 0 )
      {
        a2 = (__int64 *)sub_6E1A20(v96);
        sub_6851C0(2360, a2);
      }
      a3[2].m128i_i8[9] |= 2u;
      v75 = 0;
    }
  }
  v91 = *(_QWORD *)(v97 + 176);
  if ( !v91 && (*(_WORD *)(v97 + 168) & 0x2080) == 0 )
  {
    v95 = 1;
    v88 = (*(_BYTE *)(v97 + 169) & 1) == 0;
    v94 = sub_8D3410(m);
    if ( !v94 )
      goto LABEL_24;
    goto LABEL_20;
  }
  if ( (unsigned int)sub_8D43F0(v97) || (v95 = sub_8D3D40(m)) != 0 || (a3[2].m128i_i8[9] & 0x20) != 0 )
  {
    v88 = 0;
    v95 = 1;
    v91 = 0;
  }
  else
  {
    v88 = 0;
    v91 = *(_QWORD *)(v97 + 176);
  }
  v94 = sub_8D3410(m);
  if ( v94 )
LABEL_20:
    v94 = sub_8D4440(m) != 0;
LABEL_24:
  v12 = (__int64)v96;
  if ( v96 )
  {
    v13 = a3;
    v14 = 0;
    v15 = 0;
    while ( *(_BYTE *)(v12 + 8) != 2 )
    {
      v16 = v13[2].m128i_i8[9];
      if ( v94 )
        goto LABEL_165;
      if ( v91 <= v15 && (v95 & 1) == 0 )
      {
        v95 = 0;
        a3 = v13;
        goto LABEL_58;
      }
      if ( v16 >= 0 && v13[2].m128i_i64[0] )
      {
        if ( (unsigned int)sub_8D3BB0(m) && v90 == 1 )
          goto LABEL_34;
        v35 = *(_BYTE *)(m + 140);
        if ( v35 == 12 )
        {
          v36 = m;
          do
          {
            v36 = *(_QWORD *)(v36 + 160);
            v35 = *(_BYTE *)(v36 + 140);
          }
          while ( v35 == 12 );
        }
        if ( !v35 )
LABEL_34:
          sub_694F90(v13[2].m128i_i64[0]);
      }
      a2 = (__int64 *)m;
      sub_634B10((__int64 *)&v96, m, 0, v13, (__int64)a4, v99);
      if ( (v13[2].m128i_i8[8] & 0x40) == 0 )
      {
        a2 = (__int64 *)*a5;
        sub_72A690(v99[0], *a5, 0, 0);
      }
      ++v15;
      v17 = 1;
      if ( (v13[2].m128i_i8[9] & 0x20) == 0 )
        v17 = v95;
      v12 = (__int64)v96;
      v95 = v17;
      if ( v14 < v15 )
        v14 = v15;
LABEL_41:
      if ( !v12 )
      {
        a3 = v13;
        goto LABEL_57;
      }
    }
    if ( v90 != 1 )
    {
      v16 = v13[2].m128i_i8[9];
      if ( (v16 & 0x40) == 0 )
      {
LABEL_165:
        a3 = v13;
        goto LABEL_58;
      }
    }
    v13[2].m128i_i8[9] &= ~0x40u;
    v18 = v97;
    v98 = (__int64 *)v12;
    for ( i = (__int64 *)*a5; *(_BYTE *)(v18 + 140) == 12; v18 = *(_QWORD *)(v18 + 160) )
      ;
    v19 = *(_QWORD *)(v12 + 24);
    if ( (*(_WORD *)(v18 + 168) & 0x180) == 0 )
    {
      v20 = *(_QWORD *)(v18 + 176);
      if ( v20 )
      {
        if ( v19 )
          goto LABEL_95;
        v33 = *(_QWORD *)(v12 + 32);
        if ( v20 <= v33 || (v34 = *(_QWORD *)(v12 + 40), v20 <= v34) )
        {
          v21 = v15;
          a3 = v13;
LABEL_93:
          a2 = (__int64 *)sub_6E1A20(v12);
          sub_6851C0(175, a2);
LABEL_52:
          if ( *v98 && *(_BYTE *)(*v98 + 8) == 3 )
            sub_6BBB10(v98);
          a3[2].m128i_i8[9] |= 2u;
          v96 = 0;
          if ( v14 < v21 )
            v14 = v21;
LABEL_57:
          v16 = a3[2].m128i_i8[9];
          goto LABEL_58;
        }
        goto LABEL_102;
      }
      if ( (*(_BYTE *)(v18 + 169) & 0x20) != 0 )
      {
        v21 = v15;
        a3 = v13;
        if ( !v19 )
          goto LABEL_93;
        goto LABEL_51;
      }
    }
    if ( v19 )
    {
LABEL_95:
      v21 = v15;
      a3 = v13;
LABEL_51:
      a2 = (__int64 *)sub_6E1A20(v12);
      sub_6851C0(1045, a2);
      goto LABEL_52;
    }
    v33 = *(_QWORD *)(v12 + 32);
    v34 = *(_QWORD *)(v12 + 40);
LABEL_102:
    v15 = v34 + 1;
    v86 = v34 + 1 - v33;
    if ( dword_4F077C4 != 2 )
      goto LABEL_103;
    v81 = v18;
    v47 = sub_8D40F0(v18);
    v18 = v81;
    v49 = v47;
    for ( j = *(_BYTE *)(v47 + 140); j == 12; j = *(_BYTE *)(v49 + 140) )
      v49 = *(_QWORD *)(v49 + 160);
    if ( (unsigned __int8)(j - 9) > 2u || (*(_BYTE *)(v49 + 177) & 0x20) != 0 )
    {
      v12 = (__int64)v98;
      goto LABEL_103;
    }
    if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
    {
      v51 = (__int64)v98;
      v12 = (__int64)v98;
      if ( *(char *)(*(_QWORD *)(*(_QWORD *)v49 + 96LL) + 178LL) < 0 )
        goto LABEL_103;
      goto LABEL_144;
    }
    v63 = *(_QWORD *)(*(_QWORD *)v49 + 96LL);
    if ( *(_QWORD *)(v63 + 8) )
    {
      v78 = v49;
      v64 = sub_879360(v63, a2, v49, v48);
      v18 = v81;
      v49 = v78;
      if ( v64 )
      {
        v51 = (__int64)v98;
LABEL_144:
        v82 = v18;
        a2 = (__int64 *)sub_6E1A20(v51);
        sub_6851C0(1563, a2);
        v12 = (__int64)v98;
        v18 = v82;
        goto LABEL_103;
      }
    }
    v51 = (__int64)v98;
    v12 = (__int64)v98;
    if ( *(_BYTE *)(v49 + 140) == 12 )
    {
      v72 = v49;
      do
        v72 = *(_QWORD *)(v72 + 160);
      while ( *(_BYTE *)(v72 + 140) == 12 );
      if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v72 + 96LL) + 24LL) )
        goto LABEL_103;
      do
        v49 = *(_QWORD *)(v49 + 160);
      while ( *(_BYTE *)(v49 + 140) == 12 );
    }
    else if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v49 + 96LL) + 24LL) )
    {
      goto LABEL_103;
    }
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v49 + 96LL) + 177LL) & 2) == 0 )
      goto LABEL_144;
LABEL_103:
    if ( *(_QWORD *)v12 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 3 )
      {
        v83 = v18;
        v74 = sub_6BBB10(v12);
        v18 = v83;
        v98 = (__int64 *)v74;
      }
      else
      {
        v98 = *(__int64 **)v12;
      }
    }
    else
    {
      v98 = 0;
    }
    v13[2].m128i_i8[9] |= 0x10u;
    if ( (v13[2].m128i_i8[8] & 0x40) != 0 )
    {
      v37 = (__int64)v98;
      if ( !v98 )
        goto LABEL_132;
    }
    else
    {
      v77 = v18;
      v80 = sub_724D50(13);
      v45 = sub_72CBE0();
      v46 = (__int64)v96;
      *(_BYTE *)(v80 + 176) &= ~1u;
      *(_QWORD *)(v80 + 128) = v45;
      *(_QWORD *)(v80 + 184) = v33;
      a2 = i;
      *(_QWORD *)(v80 + 64) = *(_QWORD *)sub_6E1A20(v46);
      sub_72A690(v80, i, 0, 0);
      *((_BYTE *)i + 170) |= 0x40u;
      v37 = (__int64)v98;
      v18 = v77;
      if ( !v98 )
      {
LABEL_132:
        v96 = 0;
        a3 = v13;
        if ( v14 < v33 )
          v14 = v33;
        goto LABEL_57;
      }
    }
    v79 = 0;
    v38 = 0;
    if ( v86 > 1 )
    {
      v76 = v18;
      v39 = v13[2].m128i_i8[9];
      v13[2].m128i_i8[9] = v39 & 0xFB;
      v38 = (v39 & 4) != 0;
      v40 = sub_6E1A20(v37);
      v37 = (__int64)v98;
      v18 = v76;
      v79 = (__int64 *)v40;
    }
    a2 = *(__int64 **)(v18 + 160);
    if ( *(_BYTE *)(v37 + 8) == 2 )
      sub_6368A0(&v98, a2, v13, v99);
    else
      sub_634B10((__int64 *)&v98, (__int64)a2, 0, v13, (__int64)a4, v99);
    if ( v86 <= 1 )
    {
      v43 = v99[0];
      if ( !v99[0] )
        goto LABEL_124;
    }
    else
    {
      v41 = v13[2].m128i_i8[9];
      if ( (v41 & 4) != 0 )
      {
        if ( (v13[2].m128i_i8[8] & 0x20) != 0 )
        {
          v13[2].m128i_i8[9] = v41 | 2;
        }
        else
        {
          a2 = v79;
          sub_6851C0(971, v79);
        }
      }
      if ( v38 )
        v13[2].m128i_i8[9] |= 4u;
      v42 = v99[0];
      if ( !v99[0] )
        goto LABEL_124;
      v43 = sub_724D50(11);
      v44 = *(_QWORD *)(v42 + 64);
      *(_QWORD *)(v43 + 176) = v42;
      *(_QWORD *)(v43 + 64) = v44;
      *(_QWORD *)(v43 + 184) = v86;
      v99[0] = v43;
      while ( 1 )
      {
        while ( (*(_QWORD *)(v42 + 168) & 0xFF0000002000LL) != 0xA0000000000LL )
        {
          if ( *(_BYTE *)(v42 + 173) == 11 )
          {
            v42 = *(_QWORD *)(v42 + 176);
            if ( v42 )
              continue;
          }
          goto LABEL_123;
        }
        v42 = *(_QWORD *)(v42 + 176);
        if ( !v42 )
          goto LABEL_123;
        if ( *(_BYTE *)(v42 + 173) == 13 )
        {
          v42 = *(_QWORD *)(v42 + 120);
          if ( *(_QWORD *)(v42 + 120) )
            break;
        }
      }
      *(_BYTE *)(v43 + 192) = 1;
    }
LABEL_123:
    a2 = i;
    sub_72A690(v43, i, 0, 0);
LABEL_124:
    v12 = (__int64)v98;
    if ( v14 < v15 )
      v14 = v15;
    v96 = v98;
    goto LABEL_41;
  }
  v16 = a3[2].m128i_i8[9];
  v14 = 0;
LABEL_58:
  if ( (v16 & 2) == 0 )
  {
    if ( (v95 & 1) == 0 && v91 > v14 )
      goto LABEL_61;
    if ( (a3[2].m128i_i8[10] & 4) == 0 )
    {
LABEL_68:
      v25 = a3[2].m128i_i64[0];
      if ( v25 )
      {
        *(_BYTE *)(v25 + 86) |= 1u;
        v26 = a3[2].m128i_i64[0];
        if ( v95 )
          *(_QWORD *)(v26 + 88) = v14;
        else
          *(_QWORD *)(v26 + 88) = v91;
      }
      goto LABEL_71;
    }
    if ( v16 < 0 && !(unsigned int)sub_8D4070(v97) || (unsigned int)sub_8DBE70(m) )
      goto LABEL_67;
    if ( v95 )
      v22 = 1;
    else
LABEL_61:
      v22 = v91 - v14;
    v23 = *a5;
    for ( k = *(_BYTE *)(m + 140); k == 12; k = *(_BYTE *)(m + 140) )
      m = *(_QWORD *)(m + 160);
    if ( k == 8 )
    {
      if ( (*(_BYTE *)(m + 169) & 2) == 0 )
        v22 *= sub_8D4490(m);
      for ( m = sub_8D40F0(m); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
        ;
    }
    if ( !v22 )
    {
      if ( (a3[2].m128i_i8[8] & 0x40) == 0 && HIDWORD(qword_4F077B4) )
        sub_62F880(v23, (__int64)a2);
      goto LABEL_67;
    }
    if ( !(unsigned int)sub_8D3AD0(m) )
      goto LABEL_206;
    for ( n = m; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v66 = *(_QWORD *)(*(_QWORD *)n + 96LL);
    if ( (*(_BYTE *)(v66 + 176) & 1) == 0
      && (*(_QWORD *)(v66 + 16) || !*(_QWORD *)(v66 + 8))
      && (!dword_4D048B8 || (*(_BYTE *)(v66 + 177) & 2) != 0) )
    {
      LODWORD(v99[0]) = 0;
      if ( !(unsigned int)sub_8D3BB0(m) )
      {
        a2 = (__int64 *)m;
        v67 = v99;
        if ( (a3[2].m128i_i8[8] & 0x20) == 0 )
          v67 = 0;
        sub_876D90(m, m, a4, 1, v67);
        if ( LODWORD(v99[0]) )
          a3[2].m128i_i8[9] |= 2u;
      }
      if ( (unsigned int)sub_8D32E0(m) || (unsigned int)sub_630FC0(m) )
        a3[2].m128i_i8[9] |= 8u;
LABEL_206:
      if ( (a3[2].m128i_i8[8] & 0x40) == 0 && HIDWORD(qword_4F077B4) )
        sub_62F880(v23, (__int64)a2);
      a3[2].m128i_i8[9] |= 0x10u;
      if ( v23 )
        *(_BYTE *)(v23 + 170) |= 0x60u;
      goto LABEL_67;
    }
    a3[2].m128i_i8[11] |= 4u;
    v68 = sub_6354B0(m, a3, a4);
    a3[2].m128i_i8[11] &= ~4u;
    v69 = v68;
    if ( (a3[2].m128i_i8[8] & 0x40) == 0 )
    {
      *(_BYTE *)(v68 + 170) |= 0x80u;
      if ( (a3[2].m128i_i32[2] & 0x48000) == 0x40000 )
      {
        a3[2].m128i_i8[9] |= 4u;
        v22 = 0;
      }
      v70 = sub_724D50(11);
      v71 = *(_QWORD *)(v69 + 64);
      *(_BYTE *)(v70 + 170) |= 0x80u;
      *(_QWORD *)(v70 + 64) = v71;
      *(_QWORD *)(v70 + 184) = v22;
      *(_QWORD *)(v70 + 176) = v69;
      sub_72A690(v70, v23, 0, 0);
    }
LABEL_67:
    if ( (a3[2].m128i_i8[9] & 2) != 0 )
      goto LABEL_71;
    goto LABEL_68;
  }
LABEL_71:
  if ( !v88 )
    goto LABEL_80;
  if ( (a3[2].m128i_i8[9] & 0x20) != 0 )
  {
    if ( dword_4F04C44 == -1 )
    {
      sub_62F730(&v97, v14, (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0);
      goto LABEL_75;
    }
  }
  else
  {
    v88 = 0;
  }
  sub_62F730(&v97, v14, v88);
LABEL_75:
  if ( *a5 )
    *(_QWORD *)(*a5 + 128) = v97;
  if ( a3[2].m128i_i8[9] >= 0 && (a3[2].m128i_i8[10] & 8) != 0 )
    *v84 = v97;
LABEL_80:
  result = (__int64)v96;
  if ( v90 != 1 )
  {
LABEL_8:
    *a1 = (__int64 *)result;
    return result;
  }
  v27 = **a1;
  if ( v27 && *(_BYTE *)(v27 + 8) == 3 )
    v27 = sub_6BBB10(*a1);
  v28 = (__int64)v96;
  *a1 = (__int64 *)v27;
  if ( !v28 || !(v95 ^ 1 | v94) )
    goto LABEL_163;
  v29 = a3[2].m128i_i8[8] & 0x20;
  if ( dword_4F077C0 )
  {
    v30 = 0;
    if ( v29 )
      goto LABEL_88;
    v52 = sub_6E1A20(v28);
    v53 = 5;
    v54 = 1162;
    v55 = v52;
    goto LABEL_162;
  }
  v30 = 1;
  if ( !v29 )
  {
    v73 = sub_6E1A20(v28);
    v53 = 8;
    v54 = 146;
    v55 = v73;
LABEL_162:
    sub_684AA0(v53, v54, v55);
LABEL_163:
    v32 = a3[2].m128i_u8[9];
    goto LABEL_164;
  }
LABEL_88:
  v31 = 2 * v30;
  v32 = v31 | a3[2].m128i_i8[9] & 0xFD;
  a3[2].m128i_i8[9] = v31 | a3[2].m128i_i8[9] & 0xFD;
LABEL_164:
  result = (32 * v75) | v32 & 0xFFFFFFDF;
  a3[2].m128i_i8[9] = result;
  return result;
}
