// Function: sub_7E9AF0
// Address: 0x7e9af0
//
_QWORD *__fastcall sub_7E9AF0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 m; // rbx
  __int64 jj; // rbx
  __int64 kk; // rbx
  char v6; // al
  int v7; // esi
  __int64 mm; // rbx
  __int64 nn; // rbx
  char v10; // al
  int v11; // ecx
  __int64 i1; // rbx
  _QWORD **v13; // rdx
  _QWORD *i2; // rax
  _QWORD *v15; // rbx
  __int64 i; // r15
  __int64 v18; // rdx
  char v19; // al
  __int64 v20; // rax
  __m128i *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 j; // rax
  __int64 v25; // rax
  __m128i *v26; // rbx
  _QWORD *k; // rbx
  __int64 v28; // rax
  _QWORD *v29; // r15
  __int64 **v30; // rbx
  __int64 v31; // r14
  __int64 n; // rbx
  __int64 ii; // rbx
  __int64 v34; // rbx
  char v35; // al
  char v36; // al
  __int64 v37; // rbx
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rdi
  char v41; // al
  _QWORD *v42; // r12
  _BYTE *v43; // r13
  __int64 v44; // rsi
  __int64 *i3; // rax
  __int64 *v46; // rdx
  char v47; // dl
  __int64 v48; // rbx
  unsigned int v49; // r15d
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r15
  __int64 v53; // rbx
  __int64 v54; // rax
  __m128i *v55; // rax
  __int64 v56; // [rsp+10h] [rbp-E0h]
  __int64 v57; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+20h] [rbp-D0h]
  char v59; // [rsp+2Fh] [rbp-C1h]
  _OWORD v60[2]; // [rsp+38h] [rbp-B8h] BYREF
  _BYTE v61[144]; // [rsp+60h] [rbp-90h] BYREF

  v1 = a1;
  v59 = *(_BYTE *)(a1 + 28);
  v58 = qword_4F04C50;
  if ( !v59 )
  {
    *(_BYTE *)(a1 - 8) |= 8u;
    v2 = 0;
    goto LABEL_3;
  }
  sub_7E18E0((__int64)v61, a1, 0);
  *(_BYTE *)(a1 - 8) |= 8u;
  if ( v59 != 17 )
  {
    v2 = 0;
    goto LABEL_3;
  }
  v2 = *(_QWORD *)(a1 + 32);
  qword_4F04C50 = a1;
  if ( (*(_BYTE *)(v2 + 89) & 4) != 0 )
    sub_7E3EE0(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL));
  for ( i = *(_QWORD *)(v2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v18 = *(_QWORD *)(i + 168);
  v19 = *(_BYTE *)(v18 + 16);
  unk_4D03F58 = 0;
  if ( (v19 & 0x40) == 0 )
  {
    if ( (v19 & 0x20) == 0 )
    {
      v23 = *(_QWORD *)(a1 + 64);
      if ( !v23 )
        goto LABEL_68;
      v21 = *(__m128i **)(a1 + 40);
      goto LABEL_154;
    }
    *(_BYTE *)(v18 + 16) = v19 & 0x3F | 0x40;
  }
  v57 = v18;
  v20 = sub_72D2E0(*(_QWORD **)(i + 160));
  v21 = sub_7E2270(v20);
  v22 = *(_QWORD *)(a1 + 40);
  unk_4D03F58 = v21;
  v21[7].m128i_i64[0] = v22;
  v23 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 40) = v21;
  if ( v23 )
  {
    if ( *(char *)(v57 + 16) >= 0 )
    {
      *(_QWORD *)(v23 + 112) = v21[7].m128i_i64[0];
      v21[7].m128i_i64[0] = v23;
      goto LABEL_62;
    }
LABEL_154:
    *(_QWORD *)(v23 + 112) = v21;
    *(_QWORD *)(a1 + 40) = v23;
LABEL_62:
    if ( (unsigned int)sub_7E4C00(v2, i) )
    {
      for ( j = i; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v25 = *(_QWORD *)(*(_QWORD *)(j + 168) + 40LL);
      if ( v25 )
        v25 = sub_8D71D0(i);
      *(_QWORD *)(v23 + 120) = v25;
    }
LABEL_68:
    v26 = *(__m128i **)(a1 + 40);
    if ( !v26 )
      goto LABEL_81;
    goto LABEL_69;
  }
  v26 = v21;
  do
  {
LABEL_69:
    if ( (v26[10].m128i_i8[10] & 0x60) == 0 && v26[11].m128i_i8[1] != 5 && (v26[-1].m128i_i8[8] & 8) == 0 )
      sub_7EC5C0(v26);
    v26 = (__m128i *)v26[7].m128i_i64[0];
  }
  while ( v26 );
  for ( k = *(_QWORD **)(a1 + 40); k; k = (_QWORD *)k[14] )
  {
    while ( 1 )
    {
      v28 = k[16];
      if ( v28 )
      {
        if ( (*(_BYTE *)(v28 + 32) & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 198LL) & 0x20) == 0 )
          break;
      }
      k = (_QWORD *)k[14];
      if ( !k )
        goto LABEL_81;
    }
    k[15] = sub_72D2E0((_QWORD *)k[15]);
  }
LABEL_81:
  if ( (*(_WORD *)(v2 + 192) & 0x402) == 2 )
  {
    if ( *(_BYTE *)(v2 + 174) == 2 && !*(_QWORD *)(v2 + 320) )
      sub_7FE940(v2, 0);
    v29 = **(_QWORD ***)(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 32LL) + 168LL);
    if ( v29 )
    {
      while ( 1 )
      {
        v30 = (__int64 **)v29[15];
        if ( v30 )
          break;
LABEL_96:
        v29 = (_QWORD *)*v29;
        if ( !v29 )
        {
          v1 = a1;
          goto LABEL_3;
        }
      }
      while ( 1 )
      {
        if ( (__int64 *)v2 != v30[1] )
          goto LABEL_86;
        v31 = (__int64)v30[4];
        v60[0] = 0u;
        sub_7E02A0((__int64 **)v30[2], (__int64)v30[3], (__int64)v29, 0, v31, 0, 0, v60, (_QWORD *)v60 + 1, 0);
        if ( (!v31 || !*(_QWORD *)(v31 + 104) && (*(_BYTE *)(v31 + 96) & 2) == 0) && v60[0] == 0 )
          goto LABEL_86;
        if ( *(_BYTE *)(v2 + 174) == 2 )
        {
          if ( (*(_BYTE *)(v2 + 205) & 0x1C) == 4 )
          {
            v51 = sub_7FDF40(v30[2], 1, 0);
            if ( (*(_BYTE *)(v2 + 192) & 8) == 0 && (*(_BYTE *)(v2 + 206) & 0x10) == 0 )
              sub_7E5350(v2, v51, v31, *(__int64 *)&v60[0], *((__int64 *)&v60[0] + 1));
          }
          v42 = *(_QWORD **)(v2 + 176);
          if ( v42 )
          {
            v56 = v2;
            do
            {
              while ( 1 )
              {
                v43 = (_BYTE *)v42[1];
                if ( (v43[205] & 0x14) == 4 )
                {
                  v44 = sub_7FDF40(v30[2], (v43[205] >> 2) & 7, 0);
                  if ( (v43[192] & 8) == 0 && (v43[206] & 0x10) == 0 )
                    break;
                }
                v42 = (_QWORD *)*v42;
                if ( !v42 )
                  goto LABEL_151;
              }
              sub_7E5350((__int64)v43, v44, v31, *(__int64 *)&v60[0], *((__int64 *)&v60[0] + 1));
              v42 = (_QWORD *)*v42;
            }
            while ( v42 );
LABEL_151:
            v2 = v56;
          }
          goto LABEL_86;
        }
        if ( (*(_BYTE *)(v2 + 192) & 8) == 0 && (*(_BYTE *)(v2 + 206) & 0x10) == 0 )
        {
          sub_7E5350(v2, (__int64)v30[2], v31, *(__int64 *)&v60[0], *((__int64 *)&v60[0] + 1));
          v30 = (__int64 **)*v30;
          if ( !v30 )
            goto LABEL_96;
        }
        else
        {
LABEL_86:
          v30 = (__int64 **)*v30;
          if ( !v30 )
            goto LABEL_96;
        }
      }
    }
  }
LABEL_3:
  for ( m = *(_QWORD *)(v1 + 96); m; m = *(_QWORD *)(m + 120) )
    sub_7EB190(m);
  if ( unk_4D03F90 )
  {
    for ( n = *(_QWORD *)(v1 + 112); n; n = *(_QWORD *)(n + 112) )
    {
      if ( (*(_BYTE *)(n + 170) & 0x60) == 0 && *(_BYTE *)(n + 177) != 5 && (*(_BYTE *)(n - 8) & 8) == 0 )
        sub_7EC5C0(n);
    }
    for ( ii = *(_QWORD *)(v1 + 104); ii; ii = *(_QWORD *)(ii + 112) )
    {
      while ( (unsigned int)sub_736DD0(ii) )
      {
        ii = *(_QWORD *)(ii + 112);
        if ( !ii )
          goto LABEL_116;
      }
      sub_7EA690(ii);
    }
LABEL_116:
    if ( v59 == 6 )
    {
      if ( unk_4D04950 )
      {
        v34 = *(_QWORD *)(v1 + 112);
        if ( v34 )
        {
          if ( !*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v34 + 40) + 32LL) + 168LL) + 168LL) )
          {
            do
            {
              if ( *(_BYTE *)(v34 + 136) == 1 && !(unsigned int)sub_8D23B0(*(_QWORD *)(v34 + 120)) )
              {
                *(_BYTE *)(v34 + 88) |= 4u;
                *(_BYTE *)(v34 + 136) = 0;
                sub_7E4C10(v34);
              }
              v34 = *(_QWORD *)(v34 + 112);
            }
            while ( v34 );
          }
        }
      }
    }
  }
  else if ( v59 == 17 )
  {
    v48 = *(_QWORD *)(v1 + 32);
    v49 = sub_7DFC20(v1);
    if ( v49 )
      v49 = 1;
    else
      sub_736A50(v48);
    sub_7E9320(v1, v48, v49);
    v50 = *(_QWORD *)(v1 + 32);
    if ( *(_BYTE *)(v50 + 174) == 1
      && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v50 + 40) + 32LL) + 168LL) + 111LL) & 2) != 0 )
    {
      sub_7FAA70(v1);
    }
  }
  for ( jj = *(_QWORD *)(v1 + 120); jj; jj = *(_QWORD *)(jj + 112) )
  {
    if ( (*(_BYTE *)(jj + 170) & 0x60) == 0 && *(_BYTE *)(jj + 177) != 5 && (*(_BYTE *)(jj - 8) & 8) == 0 )
      sub_7EC5C0(jj);
  }
  for ( kk = *(_QWORD *)(v1 + 136); kk; kk = *(_QWORD *)(kk + 112) )
  {
    while ( 1 )
    {
      v6 = *(_BYTE *)(kk - 8);
      if ( (v6 & 8) == 0 )
        break;
      kk = *(_QWORD *)(kk + 112);
      if ( !kk )
        goto LABEL_22;
    }
    v7 = *(_DWORD *)(kk + 64);
    *(_BYTE *)(kk - 8) = v6 | 8;
    if ( v7 )
      *(_QWORD *)dword_4F07508 = *(_QWORD *)(kk + 64);
    if ( (*(_BYTE *)(kk + 88) & 0x70) == 0x20 )
      *(_BYTE *)(kk + 88) = *(_BYTE *)(kk + 88) & 0x8F | 0x30;
    sub_7E2D70(kk);
  }
LABEL_22:
  for ( mm = *(_QWORD *)(v1 + 144); mm; mm = *(_QWORD *)(mm + 112) )
  {
    while ( (*(_DWORD *)(mm + 192) & 0x8000400) != 0 )
    {
      mm = *(_QWORD *)(mm + 112);
      if ( !mm )
        goto LABEL_27;
    }
    sub_7F4410(mm);
  }
LABEL_27:
  for ( nn = *(_QWORD *)(v1 + 152); nn; nn = *(_QWORD *)(nn + 112) )
  {
    while ( 1 )
    {
      v10 = *(_BYTE *)(nn - 8);
      if ( (v10 & 8) == 0 )
        break;
      nn = *(_QWORD *)(nn + 112);
      if ( !nn )
        goto LABEL_36;
    }
    v11 = *(_DWORD *)(nn + 64);
    *(_BYTE *)(nn - 8) = v10 | 8;
    if ( v11 )
      *(_QWORD *)dword_4F07508 = *(_QWORD *)(nn + 64);
    if ( (*(_BYTE *)(nn + 88) & 0x70) == 0x20 )
      *(_BYTE *)(nn + 88) = *(_BYTE *)(nn + 88) & 0x8F | 0x30;
    sub_7E2D70(nn);
    sub_7EB190(*(_QWORD *)(nn + 120));
  }
LABEL_36:
  for ( i1 = *(_QWORD *)(v1 + 168); i1; i1 = *(_QWORD *)(i1 + 112) )
  {
    while ( (*(_BYTE *)(i1 + 124) & 1) != 0 )
    {
      i1 = *(_QWORD *)(i1 + 112);
      if ( !i1 )
        goto LABEL_41;
    }
    sub_7E9AF0(*(_QWORD *)(i1 + 128));
  }
LABEL_41:
  v13 = *(_QWORD ***)(v1 + 176);
  if ( v13 )
  {
    for ( i2 = *v13; i2; i2 = (_QWORD *)*i2 )
    {
      *v13 = 0;
      v13 = (_QWORD **)i2;
    }
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 184) = 0;
  }
  if ( v59 == 17 )
  {
    sub_7F4BC0(*(_QWORD *)(v1 + 160));
    if ( (*(_BYTE *)(v2 + 89) & 4) != 0 )
    {
      v35 = *(_BYTE *)(v2 + 174);
      if ( v35 == 1 || v35 == 2 )
      {
        v52 = *(_QWORD *)(v1 + 32);
        v53 = *(_QWORD *)(v1 + 40);
        sub_7FA1F0(v52);
        if ( (((*(_BYTE *)(v52 + 205) & 0x1C) - 8) & 0xF4) == 0
          && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v52 + 40) + 32LL) + 176LL) & 0x10) != 0 )
        {
          v54 = sub_7E1DF0();
          v55 = sub_7E2270(v54);
          v55[7].m128i_i64[0] = *(_QWORD *)(v53 + 112);
          *(_QWORD *)(v53 + 112) = v55;
        }
      }
    }
    sub_7E9AD0();
    v36 = *(_BYTE *)(v2 + 174);
    if ( v36 == 1 )
    {
      sub_806740(v1);
    }
    else if ( v36 == 2 )
    {
      sub_806C20(v1);
    }
    else
    {
      v37 = *(_QWORD *)(v1 + 80);
      if ( *(_BYTE *)(v37 + 40) == 19 )
        sub_7E7150(*(__m128i **)(v1 + 80), (__int64)v60 + 8, (__m128i **)v60);
      sub_7EC960(v37);
      if ( unk_4D045B0
        && *(_QWORD *)(qword_4F04C50 + 64LL)
        && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 174LL) - 1) > 1u
        && *(_BYTE *)(v37 + 40) == 11 )
      {
        v38 = sub_7E2C20(v37);
        v39 = v38;
        if ( v38 )
        {
          v40 = v38;
          while ( 1 )
          {
            v41 = *(_BYTE *)(v40 + 40);
            if ( v41 != 11 )
              break;
            v40 = sub_7E2C20(v40);
            if ( !v40 )
              goto LABEL_141;
          }
          if ( v41 == 8 )
            goto LABEL_99;
LABEL_141:
          sub_7E1720(v39, (__int64)v60 + 8);
        }
        else
        {
          sub_7E1740(v37, (__int64)v60 + 8);
        }
        sub_825C40((char *)v60 + 8, **(_QWORD **)(v37 + 80));
      }
    }
LABEL_99:
    sub_7E17F0(qword_4D03F60);
    qword_4D03F60 = 0;
    unk_4D03F58 = 0;
    if ( dword_4D04380 && *(char *)(v2 + 192) < 0 )
      sub_76FD50((_QWORD *)v1);
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 174) - 1) <= 1u && !*(_QWORD *)(v2 + 320) )
      sub_7FE940(v2, 1);
    v15 = *(_QWORD **)(v1 + 200);
    if ( !v15 )
      goto LABEL_104;
  }
  else
  {
    v15 = *(_QWORD **)(v1 + 200);
    if ( !v15 )
      goto LABEL_49;
  }
  do
  {
    sub_7EC360(v15[1], v15 + 2, v15 + 3);
    v15 = (_QWORD *)*v15;
  }
  while ( v15 );
  if ( v59 == 17 )
  {
    for ( i3 = *(__int64 **)(v1 + 200); i3; i3 = (__int64 *)*i3 )
    {
      v47 = *((_BYTE *)i3 + 16);
      if ( v47 == 1 )
      {
        v15 = i3;
      }
      else
      {
        if ( v47 != 2 )
          sub_721090();
        v46 = (__int64 *)*i3;
        if ( v15 )
          *v15 = v46;
        else
          *(_QWORD *)(v1 + 200) = v46;
      }
    }
    goto LABEL_104;
  }
LABEL_49:
  if ( v59 )
LABEL_104:
    sub_7E1AA0();
  qword_4F04C50 = v58;
  return &qword_4F04C50;
}
