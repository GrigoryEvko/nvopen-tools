// Function: sub_779600
// Address: 0x779600
//
__int64 __fastcall sub_779600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  unsigned __int8 v9; // al
  __int64 result; // rax
  char v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // r13
  unsigned int i; // edx
  __int64 v15; // rax
  __m128i *v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 j; // rdx
  unsigned __int64 v21; // r13
  unsigned int v22; // edx
  __int64 v23; // rax
  __int64 v24; // rdi
  __m128i *v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 k; // rdx
  unsigned int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // r13
  char v35; // r12
  int v36; // ebx
  int v37; // r15d
  char v38; // r15
  char v39; // al
  unsigned __int64 v40; // r14
  __int64 v41; // r15
  char v42; // al
  __int64 v43; // r9
  unsigned __int64 v44; // r12
  __int64 v45; // r14
  __m128i *v46; // r13
  int v47; // eax
  unsigned int v48; // edi
  __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 v51; // [rsp+0h] [rbp-A0h]
  __int64 v52; // [rsp+10h] [rbp-90h]
  unsigned __int64 v53; // [rsp+10h] [rbp-90h]
  unsigned __int64 v54; // [rsp+18h] [rbp-88h]
  unsigned int v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+20h] [rbp-80h]
  __int64 v57; // [rsp+20h] [rbp-80h]
  int v58; // [rsp+20h] [rbp-80h]
  __int64 v62; // [rsp+38h] [rbp-68h]
  unsigned int v63; // [rsp+48h] [rbp-58h] BYREF
  int v64; // [rsp+4Ch] [rbp-54h] BYREF
  __m128i v65; // [rsp+50h] [rbp-50h] BYREF
  _WORD v66[32]; // [rsp+60h] [rbp-40h] BYREF

  v6 = a2;
  v9 = *(_BYTE *)(a2 + 140);
  v63 = 1;
  if ( (v9 & 0xFB) == 8 )
  {
    if ( (sub_8D4C10(a2, dword_4F077C4 != 2) & 2) != 0 )
    {
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC12u, a1 + 112, a2, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    }
    while ( 1 )
    {
      v9 = *(_BYTE *)(v6 + 140);
      if ( v9 != 12 )
        break;
      v6 = *(_QWORD *)(v6 + 160);
    }
    if ( v9 > 0x13u )
    {
LABEL_13:
      v63 = 0;
      sub_721090();
    }
  }
  switch ( v9 )
  {
    case 0u:
      *(_BYTE *)(a1 + 132) |= 0x40u;
      goto LABEL_4;
    case 2u:
      sub_620DE0(&v65, 0);
      v33 = *(_QWORD *)(v6 + 128);
      if ( v33 )
      {
        v57 = v6;
        v51 = a6;
        v34 = 0;
        v35 = 1;
        v52 = a3;
        while ( 1 )
        {
          v36 = v34;
          v37 = v34;
          if ( *(_BYTE *)(a4 + v34) != 0xFF )
            v35 = 0;
          if ( !unk_4F07580 )
            v37 = v33 - 1 - v34;
          sub_620DE0(v66, *(unsigned __int8 *)(v52 + v34));
          sub_621410((__int64)v66, 8 * v37, &v64);
          if ( v64 )
            break;
          ++v34;
          sub_6213B0((__int64)&v65, (__int64)v66);
          v33 = *(_QWORD *)(v57 + 128);
          if ( (unsigned int)(v36 + 1) >= v33 )
          {
            v39 = v35;
            v6 = v57;
            a6 = v51;
            v11 = v39 & 1;
            goto LABEL_71;
          }
        }
        v38 = v35;
        v6 = v57;
        a6 = v51;
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_686CA0(0xA93u, a1 + 112, v57, (_QWORD *)(a1 + 96));
          sub_770D30(a1);
        }
        v63 = 0;
        v11 = v38 & 1;
      }
      else
      {
        v11 = 1;
      }
LABEL_71:
      if ( byte_4B6DF90[*(unsigned __int8 *)(v6 + 160)] )
        sub_6215A0(v65.m128i_i16, *(_DWORD *)(v6 + 128) * dword_4F06BA0);
      *a5 = _mm_loadu_si128(&v65);
      goto LABEL_48;
    case 3u:
      if ( *(_QWORD *)(v6 + 128) )
      {
        v32 = 0;
        v11 = 1;
        do
        {
          if ( *(_BYTE *)(a4 + v32) != 0xFF )
            v11 = 0;
          a5->m128i_i8[v32] = *(_BYTE *)(a3 + v32);
          ++v32;
        }
        while ( (unsigned __int64)(unsigned int)v32 < *(_QWORD *)(v6 + 128) );
      }
      else
      {
        v11 = 1;
      }
      goto LABEL_48;
    case 4u:
    case 5u:
    case 0xFu:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC0Fu, a1 + 112, v6, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    case 6u:
    case 7u:
    case 0xBu:
    case 0xDu:
    case 0x13u:
LABEL_4:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC13u, a1 + 112, v6, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    case 8u:
      v40 = v6;
      v41 = 1;
      break;
    case 9u:
    case 0xAu:
      sub_7764B0(a1, v6, &v63);
      result = v63;
      if ( !v63 )
        return result;
      v11 = 1;
      v12 = sub_76FF70(*(_QWORD *)(v6 + 160));
      if ( !v12 )
        goto LABEL_34;
      v56 = v6;
      v13 = v12;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v13 + 144) & 0x50) != 0x40 )
        {
          if ( (*(_BYTE *)(v13 + 144) & 4) != 0 )
          {
            v49 = a1 + 112;
            if ( *(_DWORD *)(v56 + 64) )
              v49 = v56 + 64;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xC10u, v49, v56, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
            return 0;
          }
          if ( (unsigned int)sub_8D2FB0(*(_QWORD *)(v13 + 120)) )
          {
            v50 = a1 + 112;
            if ( *(_DWORD *)(v56 + 64) )
              v50 = v56 + 64;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xC11u, v50, v56, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
            return 0;
          }
          for ( i = qword_4F08388 & (v13 >> 3); ; i = qword_4F08388 & (i + 1) )
          {
            v15 = qword_4F08380 + 16LL * i;
            if ( *(_QWORD *)v15 == v13 )
            {
              v16 = (__m128i *)((char *)a5 + *(unsigned int *)(v15 + 8));
              goto LABEL_27;
            }
            if ( !*(_QWORD *)v15 )
              break;
          }
          v16 = a5;
LABEL_27:
          v17 = *(_QWORD *)(v13 + 128);
          v18 = *(_QWORD *)(v13 + 120);
          v19 = a4 + v17;
          for ( j = a3 + v17; *(_BYTE *)(v18 + 140) == 12; v18 = *(_QWORD *)(v18 + 160) )
            ;
          if ( !(unsigned int)sub_779600(a1, v18, j, v19, v16) )
            return 0;
          if ( ((unsigned __int8)(1 << (((_BYTE)v16 - a6) & 7))
              & *(_BYTE *)(a6 + -(((unsigned int)((_DWORD)v16 - a6) >> 3) + 10))) == 0 )
            v11 = 0;
        }
        v13 = sub_76FF70(*(_QWORD *)(v13 + 112));
        if ( !v13 )
        {
          v6 = v56;
          if ( !v63 )
            return 0;
LABEL_34:
          if ( **(_QWORD **)(v6 + 168) )
          {
            v54 = v6;
            v21 = **(_QWORD **)(v6 + 168);
            do
            {
              v22 = qword_4F08388 & (v21 >> 3);
              v23 = qword_4F08380 + 16LL * v22;
              v24 = *(_QWORD *)v23;
              if ( v21 == *(_QWORD *)v23 )
              {
LABEL_78:
                v25 = (__m128i *)((char *)a5 + *(unsigned int *)(v23 + 8));
              }
              else
              {
                while ( v24 )
                {
                  v22 = qword_4F08388 & (v22 + 1);
                  v23 = qword_4F08380 + 16LL * v22;
                  v24 = *(_QWORD *)v23;
                  if ( *(_QWORD *)v23 == v21 )
                    goto LABEL_78;
                }
                v25 = a5;
              }
              v26 = *(_QWORD *)(v21 + 104);
              v27 = *(_QWORD *)(v21 + 40);
              v28 = a4 + v26;
              for ( k = a3 + v26; *(_BYTE *)(v27 + 140) == 12; v27 = *(_QWORD *)(v27 + 160) )
                ;
              if ( !(unsigned int)sub_779600(a1, v27, k, v28, v25) )
                return 0;
              v21 = *(_QWORD *)v21;
              if ( ((unsigned __int8)(1 << (((_BYTE)v25 - a6) & 7))
                  & *(_BYTE *)(a6 + -(((unsigned int)((_DWORD)v25 - a6) >> 3) + 10))) == 0 )
                v11 = 0;
            }
            while ( v21 );
            v6 = v54;
          }
LABEL_48:
          v30 = v63;
          result = v63;
          if ( v63 && v11 )
          {
LABEL_50:
            v31 = -(((unsigned int)((_DWORD)a5 - a6) >> 3) + 10);
            *(_BYTE *)(a6 + v31) |= 1 << (((_BYTE)a5 - a6) & 7);
            if ( a5 == (__m128i *)a6 )
              a5[-1].m128i_i8[7] |= 1u;
            if ( (unsigned __int8)(*(_BYTE *)(v6 + 140) - 8) > 3u )
            {
              return v30;
            }
            else
            {
              sub_778FE0(a1, (int)a5, v6, a6);
              return v63;
            }
          }
          return result;
        }
      }
    default:
      goto LABEL_13;
  }
  do
  {
    if ( (*(_BYTE *)(v40 + 169) & 1) != 0 )
    {
      v48 = 2704;
      goto LABEL_97;
    }
    if ( *(char *)(v40 + 168) < 0 )
    {
      v48 = 2999;
LABEL_97:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(v48, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    }
    v41 *= *(_QWORD *)(v40 + 176);
    do
    {
      v40 = *(_QWORD *)(v40 + 160);
      v42 = *(_BYTE *)(v40 + 140);
    }
    while ( v42 == 12 );
  }
  while ( v42 == 8 );
  if ( (unsigned __int8)(*(_BYTE *)(v40 + 140) - 2) > 1u )
    v55 = sub_7764B0(a1, v40, &v63);
  else
    v55 = 16;
  v30 = v63;
  result = v63;
  if ( !v63 )
    return result;
  v58 = 1;
  if ( !v41 )
    goto LABEL_50;
  v43 = a6;
  v44 = v40;
  v45 = a4;
  v53 = v6;
  v46 = a5;
  while ( 1 )
  {
    v62 = v43;
    if ( !(unsigned int)sub_779600(a1, v44, a3, v45, v46) )
      return 0;
    v43 = v62;
    v47 = 0;
    if ( ((unsigned __int8)(1 << (((_BYTE)v46 - v62) & 7))
        & *(_BYTE *)(v62 + -(((unsigned int)((_DWORD)v46 - v62) >> 3) + 10))) != 0 )
      v47 = v58;
    a3 += *(_QWORD *)(v44 + 128);
    v45 += *(_QWORD *)(v44 + 128);
    v58 = v47;
    v46 = (__m128i *)((char *)v46 + v55);
    if ( !--v41 )
    {
      v6 = v53;
      a6 = v62;
      v11 = v47 & 1;
      goto LABEL_48;
    }
  }
}
