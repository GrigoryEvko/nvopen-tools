// Function: sub_2F7D880
// Address: 0x2f7d880
//
__int64 __fastcall sub_2F7D880(__int64 a1)
{
  __int64 *v1; // r12
  __int64 *v2; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // r8
  __int64 v5; // rax
  unsigned __int64 v6; // r15
  int v7; // eax
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // r12
  __int64 v11; // r14
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rcx
  int v15; // r12d
  int v16; // eax
  int v17; // r9d
  unsigned int v18; // edx
  __int64 *v19; // rax
  unsigned int v20; // edx
  unsigned int v21; // eax
  __int64 *v22; // rax
  __int64 *v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 *v28; // rbx
  __int64 v29; // rdi
  __int64 v31; // rax
  __m128i *v32; // rcx
  unsigned int v33; // eax
  unsigned int v34; // esi
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rax
  unsigned int v38; // eax
  unsigned int v39; // r12d
  char v40; // al
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // rdx
  __int64 *v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rdx
  __int64 *v47; // [rsp+8h] [rbp-198h]
  int v48; // [rsp+2Ch] [rbp-174h] BYREF
  __m128i *v49; // [rsp+30h] [rbp-170h] BYREF
  __m128i *v50; // [rsp+38h] [rbp-168h] BYREF
  __int64 v51; // [rsp+40h] [rbp-160h] BYREF
  __int128 v52; // [rsp+48h] [rbp-158h] BYREF
  char v53; // [rsp+58h] [rbp-148h]
  __int64 v54; // [rsp+60h] [rbp-140h]
  __int64 *v55; // [rsp+70h] [rbp-130h] BYREF
  __int64 v56; // [rsp+78h] [rbp-128h]
  _BYTE v57[64]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v58; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+C8h] [rbp-D8h]
  __int64 *v60; // [rsp+D0h] [rbp-D0h] BYREF
  unsigned int v61; // [rsp+D8h] [rbp-C8h]
  _BYTE v62[48]; // [rsp+170h] [rbp-30h] BYREF

  v55 = (__int64 *)v57;
  v56 = 0x800000000LL;
  v2 = (__int64 *)&v60;
  v58 = 0;
  v59 = 1;
  do
  {
    *v2 = 0;
    v2 += 5;
    *((_BYTE *)v2 - 16) = 0;
    *(v2 - 1) = 0;
  }
  while ( v2 != (__int64 *)v62 );
  v3 = a1 + 48;
  v4 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v4 )
    BUG();
  v5 = *(_QWORD *)v4;
  v6 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)v4 & 4) == 0 && (*(_BYTE *)(v4 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v37 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = v37;
      if ( (*(_BYTE *)(v37 + 44) & 4) == 0 )
        break;
      v5 = *(_QWORD *)v37;
    }
  }
  v7 = 0;
  if ( v3 != v6 )
  {
    while ( 1 )
    {
      if ( (unsigned __int16)(*(_WORD *)(v6 + 68) - 14) > 1u )
      {
        ++v58;
        v21 = (unsigned int)v59 >> 1;
        if ( !((unsigned int)v59 >> 1) && !HIDWORD(v59) )
          goto LABEL_32;
        if ( (v59 & 1) != 0 )
        {
          v23 = (__int64 *)v62;
          v22 = (__int64 *)&v60;
          goto LABEL_30;
        }
        if ( 4 * v21 >= v61 || v61 <= 0x40 )
        {
          v22 = v60;
          v23 = &v60[5 * v61];
          if ( v60 != v23 )
          {
            do
            {
LABEL_30:
              *v22 = 0;
              v22 += 5;
              *((_BYTE *)v22 - 16) = 0;
              *(v22 - 1) = 0;
            }
            while ( v22 != v23 );
          }
          v59 &= 1u;
          goto LABEL_32;
        }
        if ( v21 && (v38 = v21 - 1) != 0 )
        {
          _BitScanReverse(&v38, v38);
          v39 = 1 << (33 - (v38 ^ 0x1F));
          if ( v39 - 5 <= 0x3A )
          {
            v39 = 64;
            sub_C7D6A0((__int64)v60, 40LL * v61, 8);
            v40 = v59;
            v41 = 2560;
            goto LABEL_78;
          }
          if ( v61 == v39 )
          {
            v59 &= 1u;
            if ( v59 )
            {
              v46 = (__int64 *)v62;
              v45 = (__int64 *)&v60;
            }
            else
            {
              v45 = v60;
              v46 = &v60[5 * v61];
            }
            do
            {
              if ( v45 )
              {
                *v45 = 0;
                *((_BYTE *)v45 + 24) = 0;
                v45[4] = 0;
              }
              v45 += 5;
            }
            while ( v45 != v46 );
            goto LABEL_32;
          }
          sub_C7D6A0((__int64)v60, 40LL * v61, 8);
          v40 = v59 | 1;
          LOBYTE(v59) = v59 | 1;
          if ( v39 > 4 )
          {
            v41 = 40LL * v39;
LABEL_78:
            LOBYTE(v59) = v40 & 0xFE;
            v42 = sub_C7D670(v41, 8);
            v61 = v39;
            v60 = (__int64 *)v42;
          }
        }
        else
        {
          sub_C7D6A0((__int64)v60, 40LL * v61, 8);
          LOBYTE(v59) = v59 | 1;
        }
        v59 &= 1u;
        if ( v59 )
        {
          v43 = (__int64 *)v62;
          v44 = (__int64 *)&v60;
          do
          {
LABEL_82:
            if ( v44 )
            {
              *v44 = 0;
              *((_BYTE *)v44 + 24) = 0;
              v44[4] = 0;
            }
            v44 += 5;
          }
          while ( v44 != v43 );
          goto LABEL_32;
        }
        v44 = v60;
        v43 = &v60[5 * v61];
        if ( v60 != v43 )
          goto LABEL_82;
        goto LABEL_32;
      }
      v8 = sub_B10CD0(v6 + 56);
      v9 = *(_BYTE *)(v8 - 16);
      if ( (v9 & 2) != 0 )
      {
        if ( *(_DWORD *)(v8 - 24) == 2 )
        {
          v31 = *(_QWORD *)(v8 - 32);
          goto LABEL_50;
        }
      }
      else if ( ((*(_WORD *)(v8 - 16) >> 6) & 0xF) == 2 )
      {
        v31 = v8 - 16 - 8LL * ((v9 >> 2) & 0xF);
LABEL_50:
        v10 = *(_QWORD *)(v31 + 8);
        goto LABEL_10;
      }
      v10 = 0;
LABEL_10:
      v11 = sub_2E891C0(v6);
      v51 = sub_2E89170(v6);
      if ( v11 )
        sub_AF47B0((__int64)&v52, *(unsigned __int64 **)(v11 + 16), *(unsigned __int64 **)(v11 + 24));
      else
        v53 = 0;
      v54 = v10;
      if ( !(unsigned __int8)sub_F38D60((__int64)&v58, (__int64)&v51, (__int64 *)&v49) )
      {
        v32 = v49;
        ++v58;
        v50 = v49;
        v33 = ((unsigned int)v59 >> 1) + 1;
        if ( (v59 & 1) == 0 )
        {
          v34 = v61;
          if ( 3 * v61 > 4 * v33 )
            goto LABEL_53;
LABEL_63:
          v34 *= 2;
          goto LABEL_64;
        }
        v34 = 4;
        if ( 4 * v33 >= 0xC )
          goto LABEL_63;
LABEL_53:
        if ( v34 - (v33 + HIDWORD(v59)) <= v34 >> 3 )
        {
LABEL_64:
          sub_F3E3C0((__int64)&v58, v34);
          sub_F38D60((__int64)&v58, (__int64)&v51, (__int64 *)&v50);
          v32 = v50;
          v33 = ((unsigned int)v59 >> 1) + 1;
        }
        LODWORD(v59) = v59 & 1 | (2 * v33);
        if ( v32->m128i_i64[0] || v32[1].m128i_i8[8] || v32[2].m128i_i64[0] )
          --HIDWORD(v59);
        *v32 = _mm_loadu_si128((const __m128i *)&v51);
        v32[1] = _mm_loadu_si128((const __m128i *)((char *)&v52 + 8));
        v32[2].m128i_i64[0] = v54;
        goto LABEL_32;
      }
      if ( *(_WORD *)(v6 + 68) == 14 && **(_BYTE **)(v6 + 32) )
      {
        if ( (v59 & 1) != 0 )
        {
          v14 = (__int64 *)&v60;
          v15 = 3;
        }
        else
        {
          v14 = v60;
          if ( !v61 )
            goto LABEL_32;
          v15 = v61 - 1;
        }
        v48 = 0;
        if ( v53 )
          v48 = WORD4(v52) | ((_DWORD)v52 << 16);
        v47 = v14;
        v50 = (__m128i *)v54;
        v49 = (__m128i *)v51;
        v16 = sub_F11290((__int64 *)&v49, &v48, (__int64 *)&v50);
        v17 = 1;
        v18 = v15 & v16;
        while ( 2 )
        {
          v19 = &v47[5 * v18];
          if ( *v19 == v51 && v53 == *((_BYTE *)v19 + 24) )
          {
            if ( !v53 || v52 == *(_OWORD *)(v19 + 1) )
            {
              if ( v54 == v19[4] )
              {
                *v19 = 0;
                v19[1] = 0;
                v19[2] = 0;
                *((_BYTE *)v19 + 24) = 1;
                v19[4] = 0;
                ++HIDWORD(v59);
                LODWORD(v59) = (2 * ((unsigned int)v59 >> 1) - 2) | v59 & 1;
                goto LABEL_32;
              }
              goto LABEL_22;
            }
          }
          else
          {
LABEL_22:
            if ( !*v19 && !*((_BYTE *)v19 + 24) && !v19[4] )
              goto LABEL_32;
          }
          v20 = v17 + v18;
          ++v17;
          v18 = v15 & v20;
          continue;
        }
      }
      v35 = (unsigned int)v56;
      v36 = (unsigned int)v56 + 1LL;
      if ( v36 > HIDWORD(v56) )
      {
        sub_C8D5F0((__int64)&v55, v57, v36, 8u, v12, v13);
        v35 = (unsigned int)v56;
      }
      v55[v35] = v6;
      LODWORD(v56) = v56 + 1;
LABEL_32:
      v24 = (_QWORD *)(*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL);
      v25 = v24;
      if ( !v24 )
        BUG();
      v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
      v26 = *v24;
      if ( (v26 & 4) == 0 && (*((_BYTE *)v25 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
          v6 = v27;
          if ( (*(_BYTE *)(v27 + 44) & 4) == 0 )
            break;
          v26 = *(_QWORD *)v27;
        }
      }
      if ( v3 == v6 )
      {
        v28 = v55;
        v1 = &v55[(unsigned int)v56];
        v7 = v56;
        if ( v1 != v55 )
        {
          do
          {
            v29 = *v28++;
            sub_2E88E20(v29);
          }
          while ( v1 != v28 );
          v7 = v56;
        }
        break;
      }
    }
  }
  LOBYTE(v1) = v7 != 0;
  if ( (v59 & 1) == 0 )
    sub_C7D6A0((__int64)v60, 40LL * v61, 8);
  if ( v55 != (__int64 *)v57 )
    _libc_free((unsigned __int64)v55);
  return (unsigned int)v1;
}
