// Function: sub_77B2E0
// Address: 0x77b2e0
//
__int64 __fastcall sub_77B2E0(__int64 a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r12
  char v10; // al
  char i; // al
  unsigned int v12; // r8d
  unsigned __int64 v14; // r15
  unsigned int j; // edx
  __int64 v16; // rax
  const __m128i *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 k; // rcx
  unsigned __int64 m; // r12
  unsigned int n; // edx
  __int64 v24; // rax
  const __m128i *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // r8
  __int64 ii; // rcx
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // r15
  int v33; // r8d
  int v34; // r14d
  __int64 v35; // r15
  char v36; // al
  unsigned int v37; // eax
  unsigned int v38; // edi
  __int64 v39; // rsi
  __int64 v40; // rsi
  int v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  unsigned int v45; // [rsp+30h] [rbp-60h] BYREF
  int v46; // [rsp+34h] [rbp-5Ch] BYREF
  __int64 v47; // [rsp+38h] [rbp-58h] BYREF
  __m128i v48; // [rsp+40h] [rbp-50h] BYREF
  __int16 v49[32]; // [rsp+50h] [rbp-40h] BYREF

  v8 = a2;
  v10 = *(_BYTE *)(a2 + 140) & 0xFB;
  v45 = 1;
  if ( v10 == 8 && (sub_8D4C10(a2, dword_4F077C4 != 2) & 2) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_686CA0(0xC12u, a6 + 28, a2, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
    return 0;
  }
  if ( dword_4F06BA0 > 8 )
  {
    if ( (dword_4F06BA0 & 7) == 0 )
      goto LABEL_4;
LABEL_8:
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xC5Eu, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
    }
    return 0;
  }
  if ( 8 % dword_4F06BA0 )
    goto LABEL_8;
LABEL_4:
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v8 + 140) )
    v8 = *(_QWORD *)(v8 + 160);
  switch ( i )
  {
    case 0:
      *(_BYTE *)(a1 + 132) |= 0x40u;
      goto LABEL_16;
    case 2:
      v31 = *(_QWORD *)(v8 + 128);
      v48 = _mm_loadu_si128(a3);
      if ( v31 )
      {
        v32 = 0;
        do
        {
          v33 = v32;
          if ( !unk_4F07580 )
            v33 = v31 - 1 - v32;
          v34 = 8 * v33;
          v41 = v32;
          sub_620DE0(v49, 0xFFu);
          sub_621410((__int64)v49, v34, &v46);
          sub_6213D0((__int64)v49, (__int64)&v48);
          sub_6214E0(v49, v34, 0, 0);
          sub_620E00(v49, 0, &v47, &v46);
          *(_BYTE *)(a4 + v32) = v47;
          *(_BYTE *)(a5 + v32++) = -1;
          v31 = *(_QWORD *)(v8 + 128);
        }
        while ( (unsigned int)(v41 + 1) < v31 );
      }
      return v45;
    case 3:
      v12 = v45;
      if ( *(_QWORD *)(v8 + 128) )
      {
        v30 = 0;
        do
        {
          *(_BYTE *)(a4 + v30) = a3->m128i_i8[v30];
          *(_BYTE *)(a5 + v30++) = -1;
        }
        while ( (unsigned __int64)(unsigned int)v30 < *(_QWORD *)(v8 + 128) );
      }
      return v12;
    case 4:
    case 5:
    case 15:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC0Fu, a6 + 28, v8, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    case 6:
    case 7:
    case 11:
    case 13:
    case 19:
LABEL_16:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC13u, a6 + 28, v8, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    case 8:
      v35 = 1;
      break;
    case 9:
    case 10:
      sub_7764B0(a1, v8, &v45);
      if ( !v45 )
        return 0;
      v14 = sub_76FF70(*(_QWORD *)(v8 + 160));
      if ( !v14 )
        goto LABEL_35;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v14 + 144) & 0x50) != 0x40 )
        {
          if ( (*(_BYTE *)(v14 + 144) & 4) != 0 )
          {
            v39 = a1 + 112;
            if ( *(_DWORD *)(v8 + 64) )
              v39 = v8 + 64;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xC10u, v39, v8, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
            return 0;
          }
          if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(v14 + 120)) )
          {
            v40 = a1 + 112;
            if ( *(_DWORD *)(v8 + 64) )
              v40 = v8 + 64;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xC11u, v40, v8, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
            return 0;
          }
          for ( j = qword_4F08388 & (v14 >> 3); ; j = qword_4F08388 & (j + 1) )
          {
            v16 = qword_4F08380 + 16LL * j;
            if ( *(_QWORD *)v16 == v14 )
              break;
            if ( !*(_QWORD *)v16 )
            {
              v17 = a3;
              goto LABEL_30;
            }
          }
          v17 = (const __m128i *)((char *)a3 + *(unsigned int *)(v16 + 8));
LABEL_30:
          v18 = *(_QWORD *)(v14 + 128);
          v19 = *(_QWORD *)(v14 + 120);
          v20 = a5 + v18;
          for ( k = a4 + v18; *(_BYTE *)(v19 + 140) == 12; v19 = *(_QWORD *)(v19 + 160) )
            ;
          if ( !(unsigned int)sub_77B2E0(a1, v19, v17, k, v20, a6) )
            return 0;
        }
        v14 = sub_76FF70(*(_QWORD *)(v14 + 112));
        if ( !v14 )
        {
          v12 = v45;
          if ( !v45 )
            return v12;
LABEL_35:
          for ( m = **(_QWORD **)(v8 + 168); m; m = *(_QWORD *)m )
          {
            for ( n = qword_4F08388 & (m >> 3); ; n = qword_4F08388 & (n + 1) )
            {
              v24 = qword_4F08380 + 16LL * n;
              if ( *(_QWORD *)v24 == m )
              {
                v25 = (const __m128i *)((char *)a3 + *(unsigned int *)(v24 + 8));
                goto LABEL_41;
              }
              if ( !*(_QWORD *)v24 )
                break;
            }
            v25 = a3;
LABEL_41:
            v26 = *(_QWORD *)(m + 104);
            v27 = *(_QWORD *)(m + 40);
            v28 = a5 + v26;
            for ( ii = a4 + v26; *(_BYTE *)(v27 + 140) == 12; v27 = *(_QWORD *)(v27 + 160) )
              ;
            if ( !(unsigned int)sub_77B2E0(a1, v27, v25, ii, v28, a6) )
              return 0;
          }
          return v45;
        }
      }
    default:
      v45 = 0;
      sub_721090();
  }
  do
  {
    if ( (*(_BYTE *)(v8 + 169) & 1) != 0 )
    {
      v38 = 2704;
      goto LABEL_74;
    }
    if ( *(char *)(v8 + 168) < 0 )
    {
      v38 = 2999;
LABEL_74:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(v38, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    }
    v35 *= *(_QWORD *)(v8 + 176);
    do
    {
      v8 = *(_QWORD *)(v8 + 160);
      v36 = *(_BYTE *)(v8 + 140);
    }
    while ( v36 == 12 );
  }
  while ( v36 == 8 );
  if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 2) > 1u )
    v37 = sub_7764B0(a1, v8, &v45);
  else
    v37 = 16;
  v12 = v45;
  if ( !v45 || !v35 )
    return v12;
  v42 = v37;
  while ( (unsigned int)sub_77B2E0(a1, v8, a3, a4, a5, a6) )
  {
    a3 = (const __m128i *)((char *)a3 + v42);
    a5 += *(_QWORD *)(v8 + 128);
    a4 += *(_QWORD *)(v8 + 128);
    if ( !--v35 )
      return v45;
  }
  return 0;
}
