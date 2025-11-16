// Function: sub_7764B0
// Address: 0x7764b0
//
__int64 __fastcall sub_7764B0(__int64 a1, unsigned __int64 a2, _DWORD *a3)
{
  __int64 v3; // r13
  _DWORD *v5; // rbx
  unsigned int v6; // r15d
  __int64 v8; // rsi
  _QWORD *v9; // rcx
  unsigned int v10; // esi
  unsigned int v11; // edx
  __int64 v12; // rax
  unsigned int v13; // eax
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rsi
  char j; // dl
  unsigned __int64 v21; // r14
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  char v26; // al
  unsigned __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // eax
  unsigned __int64 v30; // r14
  unsigned int v31; // r15d
  unsigned int v32; // esi
  __int64 v33; // rdi
  unsigned int v34; // ecx
  __m128i *v35; // rdx
  __m128i v36; // xmm0
  __m128i *v37; // rdx
  unsigned int v38; // r14d
  __m128i *v39; // rdx
  __m128i v40; // xmm0
  __m128i *v41; // rdx
  __int64 v42; // rsi
  unsigned int v43; // edi
  FILE *v44; // rsi
  _QWORD *v45; // rcx
  unsigned __int64 v46; // r14
  __int64 v47; // rcx
  __int64 v48; // r9
  unsigned int v49; // r13d
  unsigned __int64 v50; // r15
  unsigned int v51; // edx
  __m128i *v52; // rax
  __m128i v53; // xmm0
  __m128i *v54; // rax
  __int64 v55; // rsi
  char v56; // al
  int v57; // edx
  _DWORD *v58; // r12
  unsigned int v59; // ebx
  unsigned __int64 v60; // r15
  __int64 v61; // r14
  unsigned int v62; // edx
  __m128i *v63; // rax
  __m128i v64; // xmm0
  __m128i *v65; // rax
  unsigned __int64 i; // rdi
  unsigned int v67; // edi
  __int64 v68; // rdx
  __int64 v69; // rsi
  __int64 v70; // rsi
  unsigned int v71; // edx
  __m128i *v72; // rcx
  __m128i v73; // xmm0
  __m128i *v74; // rcx
  unsigned __int64 v75; // rdi
  unsigned int v76; // edx
  __m128i *v77; // rax
  __m128i v78; // xmm0
  __m128i *v79; // rax
  unsigned int v80; // ecx
  unsigned int v81; // edx
  __int64 v82; // rsi
  __m128i *v83; // rax
  __m128i v84; // xmm0
  __m128i *v85; // rax
  __int64 v86; // rsi
  int v87; // edx
  __int64 v88; // rsi
  unsigned __int64 v89; // [rsp+8h] [rbp-58h]
  bool v90; // [rsp+17h] [rbp-49h]
  unsigned __int64 v91; // [rsp+20h] [rbp-40h]
  unsigned __int64 v92; // [rsp+28h] [rbp-38h]
  unsigned __int64 v93; // [rsp+28h] [rbp-38h]
  unsigned __int64 v94; // [rsp+28h] [rbp-38h]

  v3 = a1;
  v5 = a3;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 140) )
    {
      case 0:
        *(_BYTE *)(a1 + 132) |= 0x40u;
        goto LABEL_11;
      case 1:
        return 0;
      case 2:
      case 3:
      case 4:
      case 0xD:
        return 16;
      case 5:
      case 6:
      case 7:
      case 0x13:
        return 32;
      case 8:
        v25 = a2;
        v23 = 1;
        do
        {
          if ( (*(_BYTE *)(v25 + 169) & 1) != 0 )
          {
            v43 = 2704;
LABEL_94:
            v44 = (FILE *)(a2 + 64);
            if ( !*(_DWORD *)(a2 + 64) )
              v44 = (FILE *)(v3 + 112);
            if ( (*(_BYTE *)(v3 + 132) & 0x20) == 0 )
            {
              sub_6855B0(v43, v44, (_QWORD *)(v3 + 96));
              sub_770D30(v3);
            }
            goto LABEL_12;
          }
          if ( *(char *)(v25 + 168) < 0 )
          {
            v43 = 2999;
            goto LABEL_94;
          }
          v23 *= *(_QWORD *)(v25 + 176);
          do
          {
            v25 = *(_QWORD *)(v25 + 160);
            v26 = *(_BYTE *)(v25 + 140);
          }
          while ( v26 == 12 );
        }
        while ( v26 == 8 );
        v6 = 16;
        if ( (unsigned __int8)(*(_BYTE *)(v25 + 140) - 2) > 1u )
          v6 = sub_7764B0(a1, v25, a3);
        if ( *v5 )
        {
          v27 = (unsigned int)&aMquoffmafNvFma[1] / v6;
          if ( v27 > 0xFFFFFF )
            v27 = 0xFFFFFF;
          if ( v23 > v27 )
          {
            v28 = a1 + 112;
            if ( *(_DWORD *)(a2 + 64) )
              v28 = a2 + 64;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xAA5u, v28, a2, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
LABEL_12:
            *v5 = 0;
            return 67108865;
          }
LABEL_101:
          v6 *= (_DWORD)v23;
        }
        return v6;
      case 9:
      case 0xA:
        v10 = qword_4F08388;
        v92 = a2 >> 3;
        v11 = qword_4F08388 & (a2 >> 3);
        while ( 2 )
        {
          v12 = qword_4F08380 + 16LL * v11;
          if ( a2 == *(_QWORD *)v12 )
          {
            v6 = *(_DWORD *)(v12 + 8);
            if ( v6 )
            {
              if ( v6 <= (unsigned int)&aMquoffmafNvFma[1] )
                return v6;
              if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
              {
                sub_686CA0(0xAF8u, a2 + 64, a2, (_QWORD *)(a1 + 96));
                sub_770D30(a1);
              }
LABEL_30:
              *v5 = 0;
              return v6;
            }
          }
          else if ( *(_QWORD *)v12 )
          {
            v11 = qword_4F08388 & (v11 + 1);
            continue;
          }
          break;
        }
        if ( (*(_DWORD *)(a2 + 176) & 0x1002000) == 0x2000 )
          goto LABEL_11;
        if ( (*(_BYTE *)(a2 + 141) & 0x20) != 0 )
        {
          v42 = a1 + 112;
          if ( *(_DWORD *)(a2 + 64) )
            v42 = a2 + 64;
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            v45 = (_QWORD *)(a1 + 96);
            goto LABEL_100;
          }
LABEL_92:
          *v5 = 0;
          return 0;
        }
        v6 = 8;
        v46 = **(_QWORD **)(a2 + 168);
        v90 = (*(_BYTE *)(a2 + 176) & 0x10) != 0;
        v48 = sub_76FF70(*(_QWORD *)(a2 + 160));
        if ( !v48 )
        {
LABEL_120:
          if ( v46 )
          {
            v91 = a2;
            v58 = v5;
            v59 = v6;
            v60 = v46;
            v89 = v46;
            do
            {
              if ( (*(_BYTE *)(v60 + 96) & 3) == 1 )
              {
                v61 = *(_QWORD *)(v60 + 40);
                if ( (v59 & 7) != 0 )
                  v59 = v59 + 8 - (v59 & 7);
                v62 = v10 & (v60 >> 3);
                v63 = (__m128i *)(v47 + 16LL * v62);
                if ( v63->m128i_i64[0] )
                {
                  v64 = _mm_loadu_si128(v63);
                  v63->m128i_i64[0] = v60;
                  v63->m128i_i32[2] = v59;
                  do
                  {
                    v62 = v10 & (v62 + 1);
                    v65 = (__m128i *)(v47 + 16LL * v62);
                  }
                  while ( v65->m128i_i64[0] );
                  *v65 = v64;
                }
                else
                {
                  v63->m128i_i64[0] = v60;
                  v63->m128i_i32[2] = v59;
                }
                ++HIDWORD(qword_4F08388);
                if ( 2 * HIDWORD(qword_4F08388) > v10 )
                  sub_7704A0((__int64)&qword_4F08380);
                if ( (unsigned __int8)(*(_BYTE *)(v61 + 140) - 2) > 1u )
                  sub_7764B0(v3, v61, v58);
                v13 = qword_4F08388;
                v47 = qword_4F08380;
                v10 = qword_4F08388;
                if ( !*v58 )
                {
                  a2 = v91;
                  goto LABEL_188;
                }
                for ( i = (unsigned __int64)(v61 + 168) >> 3; ; LODWORD(i) = v67 + 1 )
                {
                  v67 = qword_4F08388 & i;
                  v68 = qword_4F08380 + 16LL * v67;
                  if ( v61 + 168 == *(_QWORD *)v68 )
                    break;
                  if ( !*(_QWORD *)v68 )
                    goto LABEL_140;
                }
                v59 += *(_DWORD *)(v68 + 8);
LABEL_140:
                if ( v59 > 0x4000000 )
                {
                  v5 = v58;
                  a2 = v91;
                  v69 = v3 + 112;
                  if ( *(_DWORD *)(v91 + 64) )
                    goto LABEL_142;
                  goto LABEL_143;
                }
              }
              v60 = *(_QWORD *)v60;
            }
            while ( v60 );
            v6 = v59;
            v46 = v89;
            v5 = v58;
            a2 = v91;
          }
          v75 = a2 + 168;
          v76 = v10 & ((a2 + 168) >> 3);
          v77 = (__m128i *)(v47 + 16LL * v76);
          if ( v77->m128i_i64[0] )
          {
            v78 = _mm_loadu_si128(v77);
            v77->m128i_i64[0] = v75;
            v77->m128i_i32[2] = v6;
            do
            {
              v76 = v10 & (v76 + 1);
              v79 = (__m128i *)(v47 + 16LL * v76);
            }
            while ( v79->m128i_i64[0] );
            *v79 = v78;
          }
          else
          {
            v77->m128i_i64[0] = v75;
            v77->m128i_i32[2] = v6;
          }
          ++HIDWORD(qword_4F08388);
          if ( 2 * HIDWORD(qword_4F08388) > v10 )
            sub_7704A0((__int64)&qword_4F08380);
          v13 = qword_4F08388;
          if ( !v90 || !v46 )
            goto LABEL_146;
          while ( 1 )
          {
            if ( (*(_BYTE *)(v46 + 96) & 2) != 0 )
            {
              v80 = qword_4F08388;
              if ( (v6 & 7) != 0 )
                v6 = v6 + 8 - (v6 & 7);
              v81 = qword_4F08388 & (v46 >> 3);
              v82 = qword_4F08380;
              v83 = (__m128i *)(qword_4F08380 + 16LL * v81);
              if ( v83->m128i_i64[0] )
              {
                v84 = _mm_loadu_si128(v83);
                v83->m128i_i64[0] = v46;
                v83->m128i_i32[2] = v6;
                do
                {
                  v81 = v80 & (v81 + 1);
                  v85 = (__m128i *)(v82 + 16LL * v81);
                }
                while ( v85->m128i_i64[0] );
                *v85 = v84;
              }
              else
              {
                v83->m128i_i64[0] = v46;
                v83->m128i_i32[2] = v6;
              }
              ++HIDWORD(qword_4F08388);
              if ( v80 < 2 * HIDWORD(qword_4F08388) )
                sub_7704A0((__int64)&qword_4F08380);
              v86 = *(_QWORD *)(v46 + 40);
              v87 = 16;
              if ( (unsigned __int8)(*(_BYTE *)(v86 + 140) - 2) > 1u )
                v87 = sub_7764B0(v3, v86, v5);
              v6 += v87;
              if ( v6 > 0x4000000 )
                break;
            }
            v46 = *(_QWORD *)v46;
            if ( !v46 )
            {
              v13 = qword_4F08388;
              goto LABEL_146;
            }
          }
          if ( *v5 )
          {
LABEL_160:
            v69 = v3 + 112;
            if ( *(_DWORD *)(a2 + 64) )
LABEL_142:
              v69 = a2 + 64;
LABEL_143:
            if ( (*(_BYTE *)(v3 + 132) & 0x20) == 0 )
            {
              sub_686CA0(0xAA5u, v69, a2, (_QWORD *)(v3 + 96));
              sub_770D30(v3);
            }
            *v5 = 0;
            v13 = qword_4F08388;
            v6 = 67108865;
            goto LABEL_146;
          }
LABEL_187:
          v13 = qword_4F08388;
LABEL_188:
          v6 = 67108865;
LABEL_146:
          v70 = qword_4F08380;
          v71 = v13 & v92;
          v72 = (__m128i *)(qword_4F08380 + 16LL * (v13 & (unsigned int)v92));
          if ( v72->m128i_i64[0] )
          {
            v73 = _mm_loadu_si128(v72);
            v72->m128i_i64[0] = a2;
            v72->m128i_i32[2] = v6;
            do
            {
              v71 = v13 & (v71 + 1);
              v74 = (__m128i *)(v70 + 16LL * v71);
            }
            while ( v74->m128i_i64[0] );
            *v74 = v73;
          }
          else
          {
            v72->m128i_i64[0] = a2;
            v72->m128i_i32[2] = v6;
          }
          goto LABEL_84;
        }
        v49 = 8;
        v50 = v48;
        break;
      case 0xB:
        v13 = qword_4F08388;
        v14 = qword_4F08380;
        v93 = a2 >> 3;
        v15 = qword_4F08388 & (a2 >> 3);
        while ( 2 )
        {
          v16 = qword_4F08380 + 16LL * v15;
          if ( a2 == *(_QWORD *)v16 )
          {
            v6 = *(_DWORD *)(v16 + 8);
            if ( v6 )
            {
              if ( v6 <= (unsigned int)&aMquoffmafNvFma[1] )
                return v6;
              v17 = (_QWORD *)(a1 + 96);
              v18 = a2 + 64;
              if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
              {
LABEL_44:
                sub_686CA0(0xAA5u, v18, a2, v17);
                sub_770D30(a1);
              }
              goto LABEL_30;
            }
          }
          else if ( *(_QWORD *)v16 )
          {
            v15 = qword_4F08388 & (v15 + 1);
            continue;
          }
          break;
        }
        if ( (*(_BYTE *)(a2 + 177) & 0x20) != 0 )
        {
LABEL_11:
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
            v9 = (_QWORD *)(a1 + 96);
            v8 = a1 + 112;
LABEL_9:
            sub_686CA0(0xAA6u, v8, a2, v9);
            sub_770D30(a1);
          }
          goto LABEL_12;
        }
        if ( (*(_BYTE *)(a2 + 141) & 0x20) != 0 )
        {
          v45 = (_QWORD *)(a1 + 96);
          v42 = a2 + 64;
          if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          {
LABEL_100:
            sub_686CA0(0xAA5u, v42, a2, v45);
            sub_770D30(a1);
          }
          goto LABEL_92;
        }
        v30 = *(_QWORD *)(a2 + 160);
        if ( !v30 )
        {
          v6 = 8;
          goto LABEL_80;
        }
        v31 = 0;
        while ( 1 )
        {
          v32 = 16;
          if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v30 + 120) + 140LL) - 2) > 1u )
            v32 = sub_7764B0(v3, *(_QWORD *)(v30 + 120), v5);
          v13 = qword_4F08388;
          v33 = qword_4F08380;
          if ( !*v5 )
            break;
          v34 = qword_4F08388 & (v30 >> 3);
          v35 = (__m128i *)(qword_4F08380 + 16LL * v34);
          if ( v35->m128i_i64[0] )
          {
            v36 = _mm_loadu_si128(v35);
            v35->m128i_i64[0] = v30;
            v35->m128i_i32[2] = 8;
            do
            {
              v34 = v13 & (v34 + 1);
              v37 = (__m128i *)(v33 + 16LL * v34);
            }
            while ( v37->m128i_i64[0] );
            *v37 = v36;
          }
          else
          {
            v35->m128i_i64[0] = v30;
            v35->m128i_i32[2] = 8;
          }
          ++HIDWORD(qword_4F08388);
          if ( 2 * HIDWORD(qword_4F08388) > v13 )
            sub_7704A0((__int64)&qword_4F08380);
          v30 = *(_QWORD *)(v30 + 112);
          if ( v31 < v32 )
            v31 = v32;
          if ( !v30 )
          {
            v6 = v31 + 8;
            if ( v6 > (unsigned int)aMquoffmafNvFma )
            {
              v88 = v3 + 112;
              if ( *(_DWORD *)(a2 + 64) )
                v88 = a2 + 64;
              if ( (*(_BYTE *)(v3 + 132) & 0x20) == 0 )
              {
                sub_686CA0(0xAA5u, v88, a2, (_QWORD *)(v3 + 96));
                sub_770D30(v3);
              }
              *v5 = 1;
              v6 = 67108865;
            }
            v13 = qword_4F08388;
            v14 = qword_4F08380;
LABEL_80:
            v38 = v13 & v93;
            v39 = (__m128i *)(v14 + 16LL * (v13 & (unsigned int)v93));
            if ( v39->m128i_i64[0] )
            {
              v40 = _mm_loadu_si128(v39);
              v39->m128i_i64[0] = a2;
              v39->m128i_i32[2] = v6;
              do
              {
                v38 = v13 & (v38 + 1);
                v41 = (__m128i *)(v14 + 16LL * v38);
              }
              while ( v41->m128i_i64[0] );
              *v41 = v40;
            }
            else
            {
              v39->m128i_i64[0] = a2;
              v39->m128i_i32[2] = v6;
            }
LABEL_84:
            ++HIDWORD(qword_4F08388);
            if ( v13 < 2 * HIDWORD(qword_4F08388) )
              sub_7704A0((__int64)&qword_4F08380);
            return v6;
          }
        }
        v14 = qword_4F08380;
        v6 = 67108865;
        goto LABEL_80;
      case 0xC:
        a2 = *(_QWORD *)(a2 + 160);
        continue;
      case 0xE:
      case 0x15:
        goto LABEL_11;
      case 0xF:
        v19 = *(_QWORD *)(a2 + 160);
        for ( j = *(_BYTE *)(v19 + 140); j == 12; j = *(_BYTE *)(v19 + 140) )
          v19 = *(_QWORD *)(v19 + 160);
        v21 = *(_QWORD *)(a2 + 128);
        v22 = *(_QWORD *)(v19 + 128);
        v6 = 16;
        if ( (unsigned __int8)(j - 2) > 1u )
        {
          v94 = *(_QWORD *)(v19 + 128);
          v29 = sub_7764B0(a1, v19, v5);
          v22 = v94;
          v6 = v29;
        }
        if ( !*v5 )
          return v6;
        v23 = v21 / v22;
        v24 = (unsigned int)&aMquoffmafNvFma[1] / v6;
        if ( v24 > 0xFFFFFF )
          v24 = 0xFFFFFF;
        if ( v23 <= v24 )
          goto LABEL_101;
        v18 = a1 + 112;
        if ( *(_DWORD *)(a2 + 64) )
          v18 = a2 + 64;
        if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
          goto LABEL_30;
        v17 = (_QWORD *)(a1 + 96);
        goto LABEL_44;
      case 0x10:
      case 0x11:
      case 0x12:
        v8 = a1 + 112;
        if ( *(_DWORD *)(a2 + 64) )
          v8 = a2 + 64;
        v9 = (_QWORD *)(a1 + 96);
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
          goto LABEL_9;
        goto LABEL_12;
      case 0x14:
        return 24;
      default:
        sub_721090();
    }
    break;
  }
  while ( (*(_BYTE *)(v50 + 144) & 0x50) == 0x40 )
  {
LABEL_118:
    v50 = sub_76FF70(*(_QWORD *)(v50 + 112));
    if ( !v50 )
    {
      v6 = v49;
      v3 = a1;
      goto LABEL_120;
    }
  }
  if ( (v49 & 7) != 0 )
    v49 = v49 + 8 - (v49 & 7);
  v51 = v10 & (v50 >> 3);
  v52 = (__m128i *)(v47 + 16LL * v51);
  if ( v52->m128i_i64[0] )
  {
    v53 = _mm_loadu_si128(v52);
    v52->m128i_i64[0] = v50;
    v52->m128i_i32[2] = v49;
    do
    {
      v51 = v10 & (v51 + 1);
      v54 = (__m128i *)(v47 + 16LL * v51);
    }
    while ( v54->m128i_i64[0] );
    *v54 = v53;
  }
  else
  {
    v52->m128i_i64[0] = v50;
    v52->m128i_i32[2] = v49;
  }
  ++HIDWORD(qword_4F08388);
  if ( 2 * HIDWORD(qword_4F08388) > v10 )
    sub_7704A0((__int64)&qword_4F08380);
  v55 = *(_QWORD *)(v50 + 120);
  v56 = *(_BYTE *)(v55 + 140);
  if ( (*(_BYTE *)(a2 + 179) & 8) != 0 && v56 == 8 )
  {
    if ( HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 )
    {
LABEL_117:
      v10 = qword_4F08388;
      goto LABEL_118;
    }
  }
  else
  {
    v57 = 16;
    if ( (unsigned __int8)(v56 - 2) <= 1u )
      goto LABEL_116;
  }
  v57 = sub_7764B0(a1, v55, v5);
LABEL_116:
  v49 += v57;
  if ( v49 > 0x4000000 )
  {
    v3 = a1;
    if ( *v5 )
      goto LABEL_160;
    goto LABEL_187;
  }
  goto LABEL_117;
}
