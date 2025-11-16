// Function: sub_7A1C60
// Address: 0x7a1c60
//
__int64 __fastcall sub_7A1C60(__int64 a1, FILE *a2, unsigned __int64 a3, int a4, __m128i *a5, __m128i *a6, char a7)
{
  __int64 result; // rax
  char i; // al
  unsigned __int64 v13; // rcx
  __int64 v14; // r8
  __m128i *v15; // r10
  int v16; // eax
  __m128i *v17; // r10
  unsigned int v18; // eax
  _BOOL4 v19; // eax
  unsigned __int64 v20; // r10
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  __m128i *v24; // rax
  int v25; // ecx
  __int64 v26; // rsi
  unsigned int j; // edx
  _QWORD *v28; // rax
  int v29; // ecx
  unsigned int v30; // edx
  unsigned int v31; // eax
  size_t v32; // rdx
  __int64 v33; // rax
  char *v34; // r10
  __int64 *v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // ecx
  __int64 v38; // rsi
  unsigned int v39; // edx
  __m128i *v40; // rax
  __m128i v41; // xmm0
  __m128i *v42; // rax
  __int64 v43; // rax
  __int64 v44; // r10
  __int64 v45; // rax
  unsigned int v46; // eax
  __int64 v47; // rdi
  __int64 v48; // rdx
  _QWORD *v49; // rax
  size_t v50; // [rsp+8h] [rbp-148h]
  size_t v51; // [rsp+8h] [rbp-148h]
  __int64 v52; // [rsp+10h] [rbp-140h]
  __m128i *v53; // [rsp+10h] [rbp-140h]
  _QWORD *v54; // [rsp+10h] [rbp-140h]
  unsigned int v55; // [rsp+10h] [rbp-140h]
  unsigned __int64 v56; // [rsp+10h] [rbp-140h]
  int v57; // [rsp+10h] [rbp-140h]
  unsigned __int64 v58; // [rsp+10h] [rbp-140h]
  unsigned __int64 v59; // [rsp+18h] [rbp-138h]
  __m128i *v60; // [rsp+18h] [rbp-138h]
  unsigned __int8 v61; // [rsp+18h] [rbp-138h]
  __int64 v62; // [rsp+18h] [rbp-138h]
  __m128i *v63; // [rsp+18h] [rbp-138h]
  unsigned __int64 v65; // [rsp+20h] [rbp-130h]
  unsigned int v67; // [rsp+3Ch] [rbp-114h] BYREF
  __int64 *v68; // [rsp+40h] [rbp-110h] BYREF
  __int128 v69; // [rsp+48h] [rbp-108h]
  __m128i *v70; // [rsp+58h] [rbp-F8h]
  __int64 v71; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v72; // [rsp+68h] [rbp-E8h]
  int v73; // [rsp+6Ch] [rbp-E4h]
  void *s; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v75; // [rsp+78h] [rbp-D8h]
  __int64 v76; // [rsp+80h] [rbp-D0h]
  int v77; // [rsp+88h] [rbp-C8h]
  __int64 v78; // [rsp+90h] [rbp-C0h]
  __m128i v79; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+D0h] [rbp-80h]
  char v81; // [rsp+E4h] [rbp-6Ch]
  char v82; // [rsp+E5h] [rbp-6Bh]
  __int64 v83; // [rsp+118h] [rbp-38h]

  v67 = 1;
  result = dword_4F07588;
  if ( dword_4F07588 )
  {
    result = 0;
    if ( !dword_4D03F94 )
    {
      if ( dword_4F08058 )
      {
        sub_771BE0((unsigned int)dword_4F08058, a2);
        dword_4F08058 = 0;
      }
      sub_774A30((__int64)&v71, a4);
      v81 = (4 * (a7 & 1)) | v81 & 0xFB;
      v80 = *(_QWORD *)&a2->_flags;
      for ( i = *(_BYTE *)(a3 + 140); i == 12; i = *(_BYTE *)(a3 + 140) )
        a3 = *(_QWORD *)(a3 + 160);
      v13 = *(_QWORD *)(a1 + 8);
      if ( v13 )
      {
        v14 = *(_QWORD *)(v13 + 120);
        *(_QWORD *)(v13 + 120) = a3;
        v59 = v13;
        v52 = v14;
        v15 = (__m128i *)sub_77A250((__int64)&v71, v13, &v67);
        *(_QWORD *)(v59 + 120) = v52;
        if ( (*(_BYTE *)(v59 + 176) & 8) == 0 && *(_BYTE *)(v59 + 136) <= 2u )
        {
          v82 |= 4u;
          if ( (*(_BYTE *)(v59 + 172) & 0x18) == 0 || dword_4F077BC && !(_DWORD)qword_4F077B4 && qword_4F077A8 )
            v81 |= 4u;
        }
        v81 |= 2u;
      }
      else
      {
        v29 = 16;
        if ( (unsigned __int8)(i - 2) > 1u )
          v29 = sub_7764B0((__int64)&v71, a3, &v67);
        if ( !v67 )
          goto LABEL_39;
        if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 8) > 3u )
        {
          v62 = 16;
          v32 = 8;
          v31 = 16;
        }
        else
        {
          v30 = (unsigned int)(v29 + 7) >> 3;
          v31 = v30 + 9;
          if ( (((_BYTE)v30 + 9) & 7) != 0 )
            v31 = v30 + 17 - (((_BYTE)v30 + 9) & 7);
          v62 = v31;
          v32 = v31 - 8LL;
        }
        v33 = v29 + v31;
        if ( (unsigned int)v33 > 0x400 )
        {
          v51 = v32;
          v57 = v33 + 16;
          v43 = sub_822B10((unsigned int)(v33 + 16));
          v32 = v51;
          v44 = v43;
          v45 = v76;
          *(_DWORD *)(v44 + 8) = v57;
          *(_QWORD *)v44 = v45;
          *(_DWORD *)(v44 + 12) = v77;
          v76 = v44;
          v34 = (char *)(v44 + 16);
        }
        else
        {
          v34 = (char *)s;
          if ( (v33 & 7) != 0 )
            v33 = (_DWORD)v33 + 8 - (unsigned int)(v33 & 7);
          if ( 0x10000 - ((int)s - (int)v75) < (unsigned int)v33 )
          {
            v50 = v32;
            v55 = v33;
            sub_772E70(&s);
            v34 = (char *)s;
            v32 = v50;
            v33 = v55;
          }
          s = &v34[v33];
        }
        v15 = (__m128i *)((char *)memset(v34, 0, v32) + v62);
        v15[-1].m128i_i64[1] = a3;
        if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) <= 2u )
          v15->m128i_i64[0] = 0;
      }
      if ( v67 )
      {
        a5[8].m128i_i64[0] = a3;
        if ( *(_BYTE *)(a1 + 48) == 1 )
        {
          v63 = v15;
          sub_7790A0((__int64)&v71, v15, a3, (__int64)v15);
          v18 = v67;
          v17 = v63;
        }
        else
        {
          v68 = (__int64 *)v15;
          v70 = v15;
          v60 = v15;
          v69 = 0;
          DWORD1(v69) = 1;
          v16 = sub_79B7D0((__int64)&v71, (const __m128i *)a1, a2, (__int64)&v68, 0, 0);
          v17 = v60;
          if ( v16 )
          {
            v18 = v67;
          }
          else
          {
            if ( (v81 & 0x40) == 0 )
            {
              v67 = 0;
              goto LABEL_43;
            }
            sub_72C970((__int64)a5);
            v18 = v67;
            v17 = v60;
          }
        }
        if ( v18 && (v81 & 0x40) == 0 )
        {
          v61 = *(_BYTE *)(a3 + 140) - 9;
          if ( v61 <= 2u )
          {
            v56 = (unsigned __int64)v17;
            v68 = sub_724DC0();
            sub_724C70((__int64)v68, 6);
            v35 = v68;
            v17 = (__m128i *)v56;
            *((_BYTE *)v68 + 176) = 1;
            v36 = *(_QWORD *)(a1 + 8);
            if ( v36 )
              v35[23] = v36;
            v37 = v72;
            v38 = v71;
            v39 = v72 & (v56 >> 3);
            v40 = (__m128i *)(v71 + 16LL * v39);
            if ( v40->m128i_i64[0] )
            {
              v41 = _mm_loadu_si128(v40);
              v40->m128i_i64[0] = v56;
              v40->m128i_i64[1] = (__int64)v35;
              do
              {
                v39 = v37 & (v39 + 1);
                v42 = (__m128i *)(v38 + 16LL * v39);
              }
              while ( v42->m128i_i64[0] );
              *v42 = v41;
            }
            else
            {
              v40->m128i_i64[0] = v56;
              v40->m128i_i64[1] = (__int64)v68;
            }
            ++v73;
            if ( 2 * v73 > v37 )
            {
              sub_7704A0((__int64)&v71);
              v17 = (__m128i *)v56;
            }
          }
          v53 = v17;
          v19 = sub_77D750((__int64)&v71, v17, (__int64)v17, a3, (__int64)a5);
          v20 = (unsigned __int64)v53;
          if ( v19 )
          {
            if ( !v78 || a4 && (v21 = sub_799890((__int64)&v71), v20 = (unsigned __int64)v53, v21) )
            {
              if ( !*(_QWORD *)(a1 + 16)
                || (v54 = (_QWORD *)v20,
                    sub_779EA0((__int64)&v71, (char *)v20, a3, v20),
                    v22 = sub_798FD0((__int64)&v71, *(__int64 **)(a1 + 16), a2, v54, (__int64)v54, 1),
                    v20 = (unsigned __int64)v54,
                    v22) )
              {
                if ( v83 )
                {
                  v65 = v20;
                  sub_773640((__int64)&v71);
                  v20 = v65;
                  v67 = 0;
                }
                else
                {
                  if ( (unsigned __int8)(*(_BYTE *)(a1 + 48) - 3) <= 1u
                    && (*(_BYTE *)(a1 + 51) & 5) == 0
                    && (v48 = *(_QWORD *)(a1 + 56), (dword_4F07270[0] == unk_4F073B8) == (*(_BYTE *)(v48 - 8) & 1)) )
                  {
                    a5[9].m128i_i64[0] = v48;
                  }
                  else
                  {
                    v58 = v20;
                    v49 = sub_726700(31);
                    v20 = v58;
                    a5[9].m128i_i64[0] = (__int64)v49;
                    *v49 = a3;
                    *(_QWORD *)(a5[9].m128i_i64[0] + 28) = *(_QWORD *)&a2->_flags;
                    *(_QWORD *)(a5[9].m128i_i64[0] + 56) = a1;
                  }
                  if ( (*(_BYTE *)(a1 + 50) & 0x10) != 0 )
                    a5[10].m128i_i8[8] |= 0x20u;
                }
                goto LABEL_29;
              }
              v23 = *(_QWORD *)(a1 + 8);
              if ( v23 && (*(_BYTE *)(v23 + 172) & 0x10) != 0 && !v83 && *(_BYTE *)(a1 + 48) > 2u )
              {
                *(_BYTE *)(a1 + 48) = 2;
                v24 = sub_740630(a5);
                *(_BYTE *)(a1 + 72) &= ~1u;
                v20 = (unsigned __int64)v54;
                *(_QWORD *)(a1 + 56) = v24;
                *(_QWORD *)(a1 + 64) = 0;
              }
            }
          }
          v67 = 0;
LABEL_29:
          if ( v61 <= 2u )
          {
            v25 = v72;
            v26 = v71;
            for ( j = v72 & (v20 >> 3); ; j = v72 & (j + 1) )
            {
              v28 = (_QWORD *)(v71 + 16LL * j);
              if ( *v28 == v20 )
                break;
            }
            *v28 = 0;
            if ( *(_QWORD *)(v26 + 16LL * ((j + 1) & v25)) )
              sub_771200(v71, v72, j);
            --v73;
            sub_724E30((__int64)&v68);
          }
        }
LABEL_43:
        *a6 = _mm_loadu_si128(&v79);
        sub_771990((__int64)&v71);
        return v67;
      }
LABEL_39:
      if ( (v81 & 0x40) != 0 )
      {
        sub_72C970((__int64)a5);
        v67 = 1;
      }
      else if ( a4 && *(_BYTE *)(a1 + 48) == 6 )
      {
        v46 = sub_7A24A0(*(_QWORD *)(a1 + 56), (v81 & 4) != 0, a6);
        v47 = *(_QWORD *)(a1 + 56);
        v67 = v46;
        sub_740190(v47, a5, 0x800u);
      }
      goto LABEL_43;
    }
  }
  return result;
}
