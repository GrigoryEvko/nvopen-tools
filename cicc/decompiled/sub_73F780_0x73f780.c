// Function: sub_73F780
// Address: 0x73f780
//
__m128i *__fastcall sub_73F780(__int64 a1, unsigned int a2, _QWORD *a3)
{
  char v5; // bl
  __m128i *v6; // r15
  char v7; // al
  char *v8; // rcx
  __int64 v9; // rdi
  unsigned int v10; // r14d
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rax
  _BYTE *v20; // rax
  _BOOL4 v21; // esi
  __int64 v22; // rax
  _QWORD *v23; // rax
  const __m128i *v24; // rdi
  __int64 *v25; // rax
  const __m128i *v26; // r14
  _QWORD *v27; // rax
  const __m128i *v28; // rdi
  __int64 v29; // rax
  int v30; // [rsp+0h] [rbp-50h]
  bool v31; // [rsp+7h] [rbp-49h]
  _QWORD *v32; // [rsp+8h] [rbp-48h]
  _QWORD *i; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v6 = (__m128i *)sub_725A70(*(_BYTE *)(a1 + 48));
  *v6 = _mm_loadu_si128((const __m128i *)a1);
  v6[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v6[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v6[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v6[4] = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v6[5] = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v6[6] = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v6[7] = _mm_loadu_si128((const __m128i *)(a1 + 112));
  if ( (*(_BYTE *)(a1 + 51) & 0xA) != 0 )
  {
    v23 = (_QWORD *)qword_4F07AD8;
    if ( qword_4F07AD8 )
      qword_4F07AD8 = *(_QWORD *)qword_4F07AD8;
    else
      v23 = (_QWORD *)sub_823970(24);
    *v23 = *a3;
    *a3 = v23;
    v23[1] = a1;
    v23[2] = v6;
  }
  if ( (a2 & 4) != 0 )
    v6[3].m128i_i8[1] |= 4u;
  if ( (a2 & 1) != 0 && *(_QWORD *)(a1 + 8) )
    v6->m128i_i64[1] = sub_76DBC0();
  v6[3].m128i_i8[2] &= ~4u;
  v6[6].m128i_i64[1] = 0;
  v7 = *(_BYTE *)(a1 + 50);
  v6[2].m128i_i64[1] = 0;
  v8 = *(char **)(a1 + 40);
  v34 = (__int64)v8;
  v31 = (v7 & 4) != 0;
  if ( !v8 || *(_BYTE *)qword_4F06BC0 == 4 )
  {
    v6[5].m128i_i64[0] = 0;
    switch ( *(_BYTE *)(a1 + 48) )
    {
      case 0:
      case 1:
        goto LABEL_17;
      case 2:
      case 6:
      case 8:
      case 9:
        v30 = 0;
        goto LABEL_11;
      case 3:
      case 4:
        v24 = *(const __m128i **)(a1 + 56);
        goto LABEL_58;
      case 5:
        v30 = 0;
        goto LABEL_64;
      case 7:
        v24 = *(const __m128i **)(a1 + 56);
        if ( !v24 )
          goto LABEL_17;
LABEL_58:
        v6[3].m128i_i64[1] = (__int64)sub_73A9D0(v24, a2, (__int64)a3);
        goto LABEL_17;
      default:
        goto LABEL_84;
    }
  }
  sub_733780(0x1Eu, (__int64)v6, 0, *v8, 0);
  v6[5].m128i_i64[0] = 0;
  switch ( *(_BYTE *)(a1 + 48) )
  {
    case 0:
    case 1:
      goto LABEL_16;
    case 2:
    case 6:
    case 8:
    case 9:
      v30 = 1;
LABEL_11:
      v9 = *(_QWORD *)(a1 + 56);
      v10 = a2 & 0xFFFFFFDF;
      if ( v9 )
        v6[3].m128i_i64[1] = sub_73FC90(v9, 0, v10, a3);
      v11 = *(_QWORD *)(a1 + 64);
      if ( v11 )
        v6[4].m128i_i64[0] = sub_740A90(v11, v10, a3);
      goto LABEL_15;
    case 3:
    case 4:
      v6[3].m128i_i64[1] = (__int64)sub_73A9D0(*(const __m128i **)(a1 + 56), a2, (__int64)a3);
      goto LABEL_16;
    case 5:
      v30 = 1;
LABEL_64:
      if ( (a2 & 0x10) != 0 )
      {
        v25 = *(__int64 **)(a1 + 56);
        if ( v25 )
          sub_728F70(*v25);
      }
      v26 = *(const __m128i **)(a1 + 64);
      if ( v26 )
      {
        v32 = sub_73A9D0(*(const __m128i **)(a1 + 64), a2, (__int64)a3);
        for ( i = v32; ; i = v27 )
        {
          v26 = (const __m128i *)v26[1].m128i_i64[0];
          if ( !v26 )
            break;
          v27 = sub_73A9D0(v26, a2, (__int64)a3);
          if ( v32 )
            i[2] = v27;
          else
            v32 = v27;
        }
      }
      else
      {
        v32 = 0;
      }
      v6[4].m128i_i64[0] = (__int64)v32;
LABEL_15:
      if ( v30 )
        goto LABEL_16;
      goto LABEL_17;
    case 7:
      v28 = *(const __m128i **)(a1 + 56);
      if ( v28 )
        v6[3].m128i_i64[1] = (__int64)sub_73A9D0(v28, a2, (__int64)a3);
LABEL_16:
      sub_733F40();
LABEL_17:
      if ( *(_QWORD *)(a1 + 24) || (a2 & 0x8000) != 0 )
      {
        v6[1].m128i_i64[1] = 0;
        v6[2].m128i_i64[0] = 0;
        if ( (a2 & 0x80u) == 0
          && ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 1) == 0 && (a2 & 0x1000) == 0 || dword_4D03F94) )
        {
          v20 = *(_BYTE **)(a1 + 24);
          v21 = 0;
          if ( v20 )
            v21 = *v20 == 3 || *v20 == 0;
          if ( (*(_BYTE *)(a1 + 50) & 1) != 0 )
            v21 = 0;
          sub_7340D0((__int64)v6, v21, 0);
          if ( v31 )
          {
            v22 = v6[2].m128i_i64[1];
            if ( v22
              || v6[3].m128i_i8[0] == 3
              && (v29 = v6[3].m128i_i64[1], *(_BYTE *)(v29 + 24) == 10)
              && (v22 = *(_QWORD *)(v29 + 64)) != 0 )
            {
              *(_QWORD *)(v22 + 40) = v6;
              v6[6].m128i_i64[1] = v22;
              v6[3].m128i_i8[2] |= 4u;
            }
          }
        }
        if ( (v5 & 0x10) != 0 )
          sub_728F70(**(_QWORD **)(a1 + 16));
      }
      if ( (v5 & 8) != 0 )
      {
        v6[5].m128i_i64[0] = *(_QWORD *)(a1 + 80);
        *(_QWORD *)(a1 + 80) = 0;
        if ( v34 )
        {
          v12 = *(_QWORD *)(v34 + 32);
          if ( *(_BYTE *)(v34 + 8) )
            sub_733650(v34);
          v13 = *(_QWORD *)(v12 + 48);
          v14 = *(_QWORD *)(v34 + 56);
          if ( v34 == v13 )
          {
            *(_QWORD *)(v12 + 48) = v14;
          }
          else
          {
            do
            {
              v15 = v13;
              v13 = *(_QWORD *)(v13 + 56);
            }
            while ( v34 != v13 );
            *(_QWORD *)(v15 + 56) = v14;
          }
        }
        sub_733B20((_QWORD *)a1);
      }
      if ( (v5 & 2) != 0 )
        *(_QWORD *)(a1 + 80) = 0;
      else
        v6[5].m128i_i64[0] = 0;
      v16 = *(_QWORD *)(a1 + 112);
      if ( !v16 )
        return v6;
      v17 = (_QWORD *)*a3;
      if ( !*a3 )
        return v6;
      break;
    default:
LABEL_84:
      sub_721090();
  }
  while ( v16 != v17[1] )
  {
    v17 = (_QWORD *)*v17;
    if ( !v17 )
      return v6;
  }
  v18 = v17[2];
  if ( v18 )
    v6[7].m128i_i64[0] = v18;
  return v6;
}
