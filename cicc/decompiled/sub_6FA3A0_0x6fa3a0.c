// Function: sub_6FA3A0
// Address: 0x6fa3a0
//
void __fastcall sub_6FA3A0(__m128i *a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // r15
  char i; // bl
  __int8 v5; // al
  __int8 v6; // cl
  __int64 v7; // rax
  char j; // dl
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int32 *v17; // r10
  int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char v22; // al
  __int64 v23; // rax
  int v24; // eax
  _QWORD *v25; // [rsp+10h] [rbp-1C0h]
  _QWORD *v26; // [rsp+10h] [rbp-1C0h]
  bool v27; // [rsp+1Fh] [rbp-1B1h]
  int v28; // [rsp+2Ch] [rbp-1A4h] BYREF
  __int64 v29; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v30; // [rsp+38h] [rbp-198h] BYREF
  _OWORD v31[9]; // [rsp+40h] [rbp-190h] BYREF
  __m128i v32; // [rsp+D0h] [rbp-100h]
  __m128i v33; // [rsp+E0h] [rbp-F0h]
  __m128i v34; // [rsp+F0h] [rbp-E0h]
  __m128i v35; // [rsp+100h] [rbp-D0h]
  __m128i v36; // [rsp+110h] [rbp-C0h]
  __m128i v37; // [rsp+120h] [rbp-B0h]
  __m128i v38; // [rsp+130h] [rbp-A0h]
  __m128i v39; // [rsp+140h] [rbp-90h]
  __m128i v40; // [rsp+150h] [rbp-80h]
  __m128i v41; // [rsp+160h] [rbp-70h]
  __m128i v42; // [rsp+170h] [rbp-60h]
  __m128i v43; // [rsp+180h] [rbp-50h]
  __m128i v44; // [rsp+190h] [rbp-40h]

  v2 = a1[1].m128i_i8[1] == 1;
  v28 = 0;
  if ( !v2 )
    return;
  v3 = a1->m128i_i64[0];
  if ( (*(_BYTE *)(a1->m128i_i64[0] + 140) & 0xFB) == 8 )
  {
    a2 = dword_4F077C4 != 2;
    for ( i = (sub_8D4C10(a1->m128i_i64[0], a2) & 2) != 0; *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
      ;
  }
  else
  {
    i = 0;
  }
  v31[0] = _mm_loadu_si128(a1);
  v5 = a1[1].m128i_i8[0];
  v31[1] = _mm_loadu_si128(a1 + 1);
  v31[2] = _mm_loadu_si128(a1 + 2);
  v31[3] = _mm_loadu_si128(a1 + 3);
  v31[4] = _mm_loadu_si128(a1 + 4);
  v31[5] = _mm_loadu_si128(a1 + 5);
  v31[6] = _mm_loadu_si128(a1 + 6);
  v31[7] = _mm_loadu_si128(a1 + 7);
  v31[8] = _mm_loadu_si128(a1 + 8);
  if ( v5 == 2 )
  {
    v32 = _mm_loadu_si128(a1 + 9);
    v33 = _mm_loadu_si128(a1 + 10);
    v34 = _mm_loadu_si128(a1 + 11);
    v35 = _mm_loadu_si128(a1 + 12);
    v36 = _mm_loadu_si128(a1 + 13);
    v37 = _mm_loadu_si128(a1 + 14);
    v38 = _mm_loadu_si128(a1 + 15);
    v39 = _mm_loadu_si128(a1 + 16);
    v40 = _mm_loadu_si128(a1 + 17);
    v41 = _mm_loadu_si128(a1 + 18);
    v42 = _mm_loadu_si128(a1 + 19);
    v43 = _mm_loadu_si128(a1 + 20);
    v44 = _mm_loadu_si128(a1 + 21);
  }
  else if ( v5 == 5 || v5 == 1 )
  {
    v32.m128i_i64[0] = a1[9].m128i_i64[0];
    if ( (a1[1].m128i_i8[4] & 0x10) != 0 )
      goto LABEL_9;
LABEL_36:
    sub_6E5A30(a1[5].m128i_i64[1], 4, 8);
    a2 = 32;
    sub_6E5A30(a1[5].m128i_i64[1], 32, 8);
    goto LABEL_9;
  }
  if ( (a1[1].m128i_i8[4] & 0x10) == 0 )
    goto LABEL_36;
LABEL_9:
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(v3) )
    sub_8AE000(v3);
  v6 = a1[1].m128i_i8[0];
  if ( v6 )
  {
    v7 = a1->m128i_i64[0];
    for ( j = *(_BYTE *)(a1->m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v7 + 140) )
      v7 = *(_QWORD *)(v7 + 160);
    if ( j )
    {
      if ( (*(_BYTE *)(v3 + 141) & 0x20) == 0 )
      {
LABEL_22:
        if ( v6 == 2 && a1[19].m128i_i8[13] == 12 && a1[20].m128i_i8[0] == 1 )
        {
          a1[1].m128i_i8[1] = 2;
          goto LABEL_15;
        }
        v27 = (a1[1].m128i_i8[4] & 0x10) != 0;
        sub_6ECC10((__int64)a1, a2);
        if ( dword_4F077C0 )
        {
          if ( unk_4F07708 )
          {
            if ( (unsigned int)sub_6E9790((__int64)a1, &v30) )
            {
              if ( (unsigned int)sub_6EA1E0(v30) )
              {
                v29 = sub_6EA380(v30, 0, 0, 1);
                if ( v29 )
                {
                  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
                    || (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0
                    || (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x800FF) == 0x80005 )
                  {
                    sub_69D070(0x76Du, &a1[4].m128i_i32[1]);
                  }
                  v28 = 1;
                  v29 = sub_740630(v29);
                  v13 = sub_73E830(v30);
                  v9 = v29;
                  *(_QWORD *)(v29 + 144) = v13;
                  *(_BYTE *)(v29 + 169) |= 4u;
                }
              }
            }
          }
        }
        v14 = sub_6F6F40(a1, 1, v9, v10, v11, v12);
        if ( v28 || (v16 = sub_6ED3D0(v14, (unsigned __int64)&v28, &v29, (__int64)a1[4].m128i_i64 + 4, v15, 0), v28) )
        {
          sub_6E5A30(a1[5].m128i_i64[1], 8, 65544);
          sub_6E6A50(v29, (__int64)a1);
          goto LABEL_15;
        }
        v17 = &a1[4].m128i_i32[1];
        if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
        {
          sub_6E68E0(0x1Cu, (__int64)a1);
          goto LABEL_15;
        }
        if ( !word_4D04898 )
        {
          if ( dword_4F077BC
            && dword_4F077C4 == 2
            && (unk_4F07778 > 201102 || dword_4F07774)
            && !(_DWORD)qword_4F077B4
            && (unsigned __int16)(word_4F06418[0] - 29) <= 1u )
          {
            v26 = v16;
            v24 = sub_731770(v16, 0);
            v16 = v26;
            if ( !v24 )
              goto LABEL_46;
            v17 = &a1[4].m128i_i32[1];
          }
          v25 = v16;
          v18 = sub_6E91E0(0x1Cu, v17);
          v16 = v25;
          if ( v18 )
          {
            sub_6E6840((__int64)a1);
            goto LABEL_15;
          }
        }
LABEL_46:
        *((_BYTE *)v16 + 26) = *((_BYTE *)v16 + 26) & 0x7F | (i << 7);
        sub_6E70E0(v16, (__int64)a1);
        if ( v27 )
          a1[1].m128i_i8[4] |= 0x10u;
        if ( (*(_BYTE *)(qword_4D03C50 + 21LL) & 2) != 0 )
          sub_6F4D20(a1, 0, 0, v19, v20, v21);
        if ( a1->m128i_i64[0] != *(_QWORD *)&v31[0] && *(_BYTE *)(v3 + 140) == 14 && *(_BYTE *)(v3 + 160) == 2 )
          a1->m128i_i64[0] = *(_QWORD *)&v31[0];
        goto LABEL_15;
      }
      if ( !(unsigned int)sub_8D2690(v3) )
      {
        v22 = *(_BYTE *)(v3 + 140);
        if ( dword_4F077C4 == 2 )
        {
          if ( (unsigned __int8)(v22 - 9) > 2u )
            goto LABEL_60;
          if ( (*(_BYTE *)(v3 + 177) & 0x20) == 0 || (*(_BYTE *)(v3 + 89) & 1) != 0 )
          {
            if ( dword_4D04964 )
              goto LABEL_60;
            v23 = 776LL * dword_4F04C64;
            if ( *(_BYTE *)(qword_4F04C68[0] + v23 + 4) != 1 || *(_BYTE *)(qword_4F04C68[0] + v23 - 772) != 8 )
              goto LABEL_60;
          }
        }
        else if ( v22 != 1 )
        {
LABEL_60:
          sub_6E5F60(&a1[4].m128i_i32[1], (FILE *)v3, 8);
          sub_6E6840((__int64)a1);
          goto LABEL_15;
        }
      }
      v6 = a1[1].m128i_i8[0];
      goto LABEL_22;
    }
  }
  sub_6E6870((__int64)a1);
LABEL_15:
  sub_6E4BC0((__int64)a1, (__int64)v31);
  a1[5].m128i_i64[1] = 0;
  sub_6E5070((__int64)a1, (__int64)v31);
  if ( !v28 )
    sub_6E26D0(2, (__int64)a1);
  *(_BYTE *)(qword_4D03C50 + 21LL) &= ~2u;
}
