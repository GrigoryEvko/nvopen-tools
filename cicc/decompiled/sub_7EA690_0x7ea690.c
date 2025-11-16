// Function: sub_7EA690
// Address: 0x7ea690
//
void __fastcall sub_7EA690(__int64 a1, __m128i *a2)
{
  __int64 v2; // rbx
  int v3; // r12d
  int v4; // r8d
  __int64 v5; // rax
  __int64 v6; // r13
  __m128i *v7; // r12
  __int64 v8; // r14
  __int64 *kk; // rax
  int v10; // r15d
  __int64 v11; // r14
  __int64 v12; // r13
  char v13; // al
  __int64 n; // rax
  __int64 v15; // r13
  __int64 v16; // rdi
  _QWORD *v17; // r14
  __int64 ii; // r14
  __int64 *jj; // r14
  unsigned __int8 v20; // al
  __int64 v21; // r14
  __int64 j; // rax
  __int64 v23; // rdx
  char v24; // al
  __int64 k; // rax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // r13
  unsigned int v29; // eax
  _QWORD *v30; // rdx
  char v31; // al
  __int64 v32; // rdi
  char v33; // al
  char v34; // al
  __int64 v35; // r12
  _QWORD *v36; // r12
  __m128i *v37; // r13
  __int64 v38; // r14
  const __m128i *i; // r12
  __m128i *v40; // rax
  __int64 v41; // rax
  _QWORD *v42; // r13
  __int64 m; // r12
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // r12
  __int16 v52; // [rsp-3Ah] [rbp-3Ah]

  if ( (*(_BYTE *)(a1 - 8) & 8) != 0 )
    return;
  v2 = a1;
  while ( 2 )
  {
    v3 = sub_736DD0(v2);
    if ( v3 )
      return;
    v4 = *(_DWORD *)(v2 + 64);
    *(_BYTE *)(v2 - 8) |= 8u;
    if ( v4 )
      *(_QWORD *)dword_4F07508 = *(_QWORD *)(v2 + 64);
    if ( (*(_BYTE *)(v2 + 88) & 0x70) == 0x20 )
      *(_BYTE *)(v2 + 88) = *(_BYTE *)(v2 + 88) & 0x8F | 0x30;
    sub_7E2D70(v2);
    switch ( *(_BYTE *)(v2 + 140) )
    {
      case 1:
      case 3:
      case 5:
      case 0x11:
      case 0x12:
        return;
      case 2:
        v34 = *(_BYTE *)(v2 + 161);
        if ( (v34 & 8) != 0 )
        {
          if ( (**(_BYTE **)(v2 + 176) & 1) != 0 )
          {
            v51 = *(_QWORD *)(v2 + 168);
            if ( (v34 & 0x10) != 0 )
            {
              sub_7E9AF0(*(_QWORD *)(v2 + 168));
            }
            else
            {
              while ( v51 )
              {
                sub_7EB190(v51);
                v51 = *(_QWORD *)(v51 + 120);
              }
            }
          }
          return;
        }
        v2 = *(_QWORD *)(v2 + 168);
        if ( !v2 )
          return;
LABEL_18:
        if ( (*(_BYTE *)(v2 - 8) & 8) != 0 )
          return;
        continue;
      case 6:
        if ( (*(_BYTE *)(v2 + 141) & 0x40) != 0 && (unsigned int)sub_8D2310(*(_QWORD *)(v2 + 160)) )
        {
          for ( i = *(const __m128i **)(v2 + 160); i[8].m128i_i8[12] == 12; i = (const __m128i *)i[10].m128i_i64[0] )
            ;
          v40 = (__m128i *)sub_7259C0(7);
          i[11].m128i_i64[0] = (__int64)v40;
          a2 = v40;
          sub_73BCD0(i, v40, 0);
          *(_BYTE *)(i[11].m128i_i64[0] - 8) &= ~8u;
        }
        goto LABEL_17;
      case 7:
        sub_7EA690(*(_QWORD *)(v2 + 160));
        v21 = *(_QWORD *)(v2 + 168);
        if ( unk_4F06968 )
          *(_BYTE *)(v21 + 16) &= ~2u;
        for ( j = v2; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v23 = *(_QWORD *)(j + 168);
        v24 = *(_BYTE *)(v23 + 16);
        if ( (v24 & 0x40) == 0 && (v24 & 0x20) != 0 )
          *(_BYTE *)(v23 + 16) = v24 & 0x3F | 0x40;
        if ( (*(_BYTE *)(v21 + 16) & 0x40) != 0 )
        {
          v49 = sub_72D2E0(*(_QWORD **)(v2 + 160));
          v50 = sub_724EF0(v49);
          *((_BYTE *)v50 - 8) &= ~8u;
          *v50 = *(_QWORD *)v21;
          *(_QWORD *)v21 = v50;
          *(_QWORD *)(v2 + 160) = sub_72CBE0();
          if ( !*(_QWORD *)(v21 + 40) )
          {
            v28 = *(_QWORD *)v21;
            v3 = 1;
            if ( !*(_QWORD *)v21 )
              goto LABEL_120;
            goto LABEL_64;
          }
          goto LABEL_55;
        }
        if ( *(_QWORD *)(v21 + 40) )
        {
LABEL_55:
          for ( k = v2; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
          v26 = *(_QWORD *)(*(_QWORD *)(k + 168) + 40LL);
          if ( v26 )
            v26 = sub_8D71D0(v2);
          v27 = sub_724EF0(v26);
          *((_BYTE *)v27 - 8) &= ~8u;
          v28 = (__int64)v27;
          if ( !(unsigned int)sub_7E4C00(*(_QWORD *)(v21 + 8), v2) )
          {
            v29 = *(_DWORD *)(v28 + 32) & 0xFFFC07FF;
            BYTE1(v29) |= 8u;
            *(_DWORD *)(v28 + 32) = v29;
          }
          v30 = *(_QWORD **)v21;
          if ( (*(_BYTE *)(v21 + 16) & 0xC0) == 0x40 )
          {
            *(_QWORD *)v28 = *v30;
            **(_QWORD **)v21 = v28;
            v28 = *(_QWORD *)v21;
            if ( !*(_QWORD *)v21 )
              goto LABEL_120;
          }
          else
          {
            *(_QWORD *)v28 = v30;
            *(_QWORD *)v21 = v28;
          }
          v3 = 1;
          do
          {
LABEL_64:
            v31 = *(_BYTE *)(v28 - 8);
            if ( (v31 & 8) == 0 )
            {
              *(_BYTE *)(v28 - 8) = v31 | 8;
              sub_7E1650((const char **)(v28 + 24));
              sub_7EA690(*(_QWORD *)(v28 + 8));
              if ( (*(_BYTE *)(v28 + 32) & 1) != 0 )
              {
                v47 = *(_QWORD *)(v21 + 8);
                if ( !v47 || (v3 = 1, (*(_BYTE *)(v47 + 198) & 0x20) == 0) )
                {
                  v3 = 1;
                  sub_7E4BD0(v28);
                }
              }
              if ( dword_4D03F8C )
              {
                v48 = *(_QWORD *)(v28 + 40);
                if ( v48 )
                  sub_7E99E0(v48);
              }
              *(_QWORD *)(v28 + 40) = 0;
            }
            v28 = *(_QWORD *)v28;
          }
          while ( v28 );
          if ( !v3 )
            goto LABEL_67;
LABEL_120:
          sub_7607C0(v2, 6);
          goto LABEL_67;
        }
        v28 = *(_QWORD *)v21;
        if ( *(_QWORD *)v21 )
          goto LABEL_64;
LABEL_67:
        v32 = *(_QWORD *)(v21 + 48);
        if ( v32 )
          sub_7E9AF0(v32);
        v33 = *(_BYTE *)(v21 + 17);
        if ( (v33 & 1) != 0 )
        {
          while ( *(_BYTE *)(v2 + 140) == 12 )
            v2 = *(_QWORD *)(v2 + 160);
          v41 = *(_QWORD *)(v2 + 168);
          v42 = *(_QWORD **)v41;
          for ( m = sub_8D46C0(*(_QWORD *)(*(_QWORD *)v41 + 8LL)); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          sub_7E3EE0(m);
          if ( unk_4F06890 )
            *(_QWORD *)(v2 + 160) = sub_72D2E0((_QWORD *)m);
        }
        else
        {
          if ( (v33 & 2) == 0 )
            return;
          while ( *(_BYTE *)(v2 + 140) == 12 )
            v2 = *(_QWORD *)(v2 + 160);
          v46 = *(_QWORD *)(v2 + 168);
          v42 = *(_QWORD **)v46;
          for ( m = sub_8D46C0(*(_QWORD *)(*(_QWORD *)v46 + 8LL)); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          sub_7E3EE0(m);
          if ( unk_4F0688C )
            *(_QWORD *)(v2 + 160) = sub_7E1C10();
        }
        if ( (*(_BYTE *)(m + 176) & 0x10) != 0 )
        {
          v44 = sub_7E1DF0();
          v45 = sub_724EF0(v44);
          *v45 = *v42;
          *v42 = v45;
        }
        return;
      case 8:
      case 0xF:
      case 0x10:
        goto LABEL_17;
      case 9:
      case 0xA:
      case 0xB:
        v10 = dword_4F07508[0];
        v52 = dword_4F07508[1];
        sub_7E3EE0(v2);
        v11 = *(_QWORD *)(v2 + 160);
        if ( !v11 )
          goto LABEL_32;
        do
        {
          while ( 1 )
          {
            v12 = v11;
            v11 = *(_QWORD *)(v11 + 112);
            v13 = *(_BYTE *)(v12 - 8);
            if ( (v13 & 8) == 0 )
            {
              *(_BYTE *)(v12 - 8) = v13 | 8;
              if ( *(_DWORD *)(v12 + 64) )
                *(_QWORD *)dword_4F07508 = *(_QWORD *)(v12 + 64);
              if ( (*(_BYTE *)(v12 + 88) & 0x70) == 0x20 )
                *(_BYTE *)(v12 + 88) = *(_BYTE *)(v12 + 88) & 0x8F | 0x30;
              sub_7E2D70(v12);
              sub_7EAF80(*(_QWORD *)(v12 + 120));
              if ( (*(_BYTE *)(v12 + 145) & 0x10) != 0 )
                break;
            }
            if ( !v11 )
              goto LABEL_32;
          }
          for ( n = *(_QWORD *)(v12 + 120); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
            ;
          *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(n + 168) + 208LL) + 178LL) |= 0x80u;
        }
        while ( v11 );
LABEL_32:
        v15 = *(_QWORD *)(v2 + 168);
        v16 = *(_QWORD *)(v15 + 152);
        *(_QWORD *)dword_4F07508 = *(_QWORD *)(v2 + 64);
        if ( v16 && (*(_BYTE *)(v16 + 29) & 0x20) == 0 )
        {
          v17 = *(_QWORD **)v15;
          if ( *(_QWORD *)v15 )
          {
            do
            {
              sub_7EA690(v17[5]);
              v17 = (_QWORD *)*v17;
            }
            while ( v17 );
            v16 = *(_QWORD *)(v15 + 152);
          }
          sub_7E9AF0(v16);
          for ( ii = *(_QWORD *)(v15 + 216); ii; ii = *(_QWORD *)(ii + 112) )
          {
            if ( !(unsigned int)sub_736DD0(ii) )
              sub_7EA690(ii);
          }
        }
        for ( jj = *(__int64 **)(v15 + 168); jj; jj = (__int64 *)*jj )
        {
          while ( 1 )
          {
            v20 = *((_BYTE *)jj + 8);
            if ( v20 != 1 )
              break;
            sub_7EB190(jj[4]);
            jj = (__int64 *)*jj;
            if ( !jj )
              goto LABEL_113;
          }
          if ( v20 <= 1u )
          {
            sub_7EA690(jj[4]);
          }
          else if ( (unsigned __int8)(v20 - 2) > 1u )
          {
LABEL_45:
            sub_721090();
          }
        }
LABEL_113:
        sub_7EA690(*(_QWORD *)(v15 + 208));
        if ( *(_BYTE *)(v2 + 140) == 9 )
          *(_BYTE *)(v2 + 140) = 10;
        dword_4F07508[0] = v10;
        LOWORD(dword_4F07508[1]) = v52;
        return;
      case 0xC:
        if ( *(char *)(v2 + 186) < 0 || (unsigned int)sub_8D2B50(v2) )
        {
          for ( kk = *(__int64 **)(v2 + 104); kk; kk = (__int64 *)*kk )
          {
            if ( *((_BYTE *)kk + 8) == 51 )
              *((_BYTE *)kk + 8) = 0;
          }
        }
LABEL_17:
        v2 = *(_QWORD *)(v2 + 160);
        goto LABEL_18;
      case 0xD:
        sub_7EA690(*(_QWORD *)(v2 + 160));
        v35 = *(_QWORD *)(v2 + 168);
        sub_7EA690(v35);
        if ( (unsigned int)sub_8D2310(v35) )
          v36 = (_QWORD *)sub_7E1D00(v35, a2);
        else
          v36 = sub_72BA30(unk_4D03F80);
        v37 = (__m128i *)sub_7259C0(*(_BYTE *)(v2 + 140));
        sub_73C230((const __m128i *)v2, v37);
        v38 = *(_QWORD *)(v2 + 112);
        sub_725570(v2, 12);
        *(_QWORD *)(v2 + 112) = v38;
        *(_QWORD *)(v2 + 160) = v36;
        *(_QWORD *)(v2 + 176) = v37;
        sub_760760((__int64)v36, 6, v2, (unsigned __int8)(*((_BYTE *)v36 + 140) - 9) <= 2u);
        if ( *(char *)(v2 - 8) < 0 )
        {
          sub_75B260((__int64)v36, 6u);
          if ( (unsigned __int8)(*((_BYTE *)v36 + 140) - 9) <= 2u )
            sub_75BF90((__int64)v36);
        }
        return;
      case 0xE:
        if ( !(unsigned int)sub_8D3EA0(v2) )
          return;
        v5 = sub_72CBE0();
LABEL_10:
        v6 = v5;
        v7 = (__m128i *)sub_7259C0(*(_BYTE *)(v2 + 140));
        sub_73C230((const __m128i *)v2, v7);
        v8 = *(_QWORD *)(v2 + 112);
        sub_725570(v2, 12);
        *(_QWORD *)(v2 + 160) = v6;
        *(_QWORD *)(v2 + 112) = v8;
        *(_QWORD *)(v2 + 176) = v7;
        return;
      case 0x13:
      case 0x14:
        v5 = sub_7E1C10();
        goto LABEL_10;
      default:
        goto LABEL_45;
    }
  }
}
