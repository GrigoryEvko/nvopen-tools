// Function: sub_710650
// Address: 0x710650
//
_DWORD *__fastcall sub_710650(
        const __m128i *a1,
        __int64 a2,
        __int64 a3,
        __m128i *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        int a9,
        _DWORD *a10,
        _DWORD *a11,
        _DWORD *a12)
{
  const __m128i *v12; // r15
  __int8 v15; // al
  _QWORD *v16; // rax
  int v17; // eax
  __int64 i; // r13
  __int64 v19; // rax
  _QWORD *v20; // rbx
  char v22; // al
  __int64 v23; // rax
  __int8 v24; // al
  __m128i *v25; // rdi
  int v26; // eax
  __int8 v27; // al
  __int64 v28; // r13
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 j; // rax
  char v32; // al
  __int64 v33; // rax
  __int8 v34; // al
  __int64 v35; // rax
  unsigned __int64 v36; // r9
  unsigned __int64 v37; // rsi
  int v38; // eax
  __int64 v41; // [rsp+38h] [rbp-88h]
  _QWORD *v42; // [rsp+40h] [rbp-80h]
  __int64 v43; // [rsp+48h] [rbp-78h]
  __int64 v44; // [rsp+50h] [rbp-70h]
  int v45; // [rsp+5Ch] [rbp-64h]
  _BOOL4 v46; // [rsp+68h] [rbp-58h] BYREF
  int v47; // [rsp+6Ch] [rbp-54h] BYREF
  __m128i *v48; // [rsp+70h] [rbp-50h] BYREF
  _QWORD *v49; // [rsp+78h] [rbp-48h] BYREF
  __int16 v50[32]; // [rsp+80h] [rbp-40h] BYREF

  v12 = a1;
  v45 = a5;
  *a10 = 0;
  if ( a12 )
  {
    *a12 = 0;
    if ( (*(_BYTE *)(a2 + 96) & 4) != 0 && (_DWORD)a6 )
    {
      *a12 = 286;
      return (_DWORD *)sub_72C970(a4);
    }
  }
  else if ( (*(_BYTE *)(a2 + 96) & 4) != 0 && (_DWORD)a6 )
  {
    sub_685360(0x11Eu, a11, *(_QWORD *)(a2 + 40));
    return (_DWORD *)sub_72C970(a4);
  }
  if ( !*(_QWORD *)(a2 + 112) || (v15 = a1[10].m128i_i8[13], v15 == 12) )
  {
    *a10 = 1;
    return a10;
  }
  else
  {
    if ( !v15 )
      return (_DWORD *)sub_72C970(a4);
    v48 = (__m128i *)sub_724DC0(a1, a2, a3, a12, a5, a6);
    v16 = (_QWORD *)a1[9].m128i_i64[0];
    a1[9].m128i_i64[0] = 0;
    v49 = v16;
    sub_72A510(a1, a4);
    if ( dword_4F07588 )
    {
      v17 = 0;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
        v17 = v45;
      v45 = v17;
    }
    for ( i = sub_8D46C0(a1[8].m128i_i64[0]); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v19 = *(_QWORD *)(a2 + 112);
    v20 = *(_QWORD **)(v19 + 16);
    v42 = v20;
    if ( (*(_BYTE *)(a2 + 96) & 2) == 0 )
      v20 = *(_QWORD **)(v19 + 8);
    if ( v20 != (_QWORD *)*v42 )
    {
      v44 = i;
      while ( 1 )
      {
        v28 = v20[2];
        if ( !v45 )
        {
LABEL_42:
          v45 = 0;
          goto LABEL_43;
        }
        v22 = *(_BYTE *)(v28 + 96);
        if ( (v22 & 2) != 0 )
        {
          if ( (v22 & 1) == 0 || (v23 = *(_QWORD *)(v28 + 112), *(_QWORD *)v23) )
          {
            if ( (unsigned int)sub_87DE40(v20[2], v44) )
              goto LABEL_28;
            goto LABEL_24;
          }
        }
        else
        {
          v23 = *(_QWORD *)(v28 + 112);
        }
        if ( !*(_BYTE *)(v23 + 25) || (unsigned int)sub_87D890(v44) )
          goto LABEL_28;
        if ( *(_BYTE *)(*(_QWORD *)(v28 + 112) + 25LL) == 1 )
        {
          if ( (unsigned int)sub_87D970(v44) )
            goto LABEL_28;
LABEL_24:
          if ( *(_BYTE *)(*(_QWORD *)(v28 + 112) + 25LL) == 1
            && dword_4F077BC
            && qword_4F077A8 <= 0x9DCFu
            && (unsigned int)sub_87E070(v28, a2) )
          {
            goto LABEL_28;
          }
        }
        if ( !a12 )
        {
          sub_685260(7u, 0x10Du, a11, *(_QWORD *)(v28 + 40));
          v45 = 0;
LABEL_43:
          v44 = *(_QWORD *)(v28 + 40);
          if ( a8 )
            goto LABEL_29;
          goto LABEL_44;
        }
        if ( !sub_67D3C0((int *)0x10D, 7, a11) )
          goto LABEL_42;
        v45 = 0;
        *a12 = 269;
LABEL_28:
        v44 = *(_QWORD *)(v28 + 40);
        if ( a8 )
          goto LABEL_29;
LABEL_44:
        if ( (unsigned int)sub_710600((__int64)a1) )
        {
          v20 = (_QWORD *)*v20;
          if ( (_QWORD *)*v42 == v20 )
          {
LABEL_46:
            i = v44;
            v12 = a1;
            break;
          }
        }
        else
        {
LABEL_29:
          if ( a4[10].m128i_i8[13] == 6 )
          {
            v35 = sub_77F710(a4, 0, 1);
            v41 = v35;
            v43 = *(_QWORD *)(v35 + 16);
            if ( v43 )
              *(_QWORD *)(v35 + 16) = sub_8E5310(v28, *(_QWORD *)(v43 + 56), v43);
            else
              *(_QWORD *)(v35 + 16) = v28;
          }
          else
          {
            v41 = 0;
            v43 = 0;
          }
          v24 = a1[10].m128i_i8[13];
          v25 = v48;
          if ( v24 == 1 )
          {
            *v48 = _mm_loadu_si128(a1);
            v25[1] = _mm_loadu_si128(a1 + 1);
            v25[2] = _mm_loadu_si128(a1 + 2);
            v25[3] = _mm_loadu_si128(a1 + 3);
            v25[4] = _mm_loadu_si128(a1 + 4);
            v25[5] = _mm_loadu_si128(a1 + 5);
            v25[6] = _mm_loadu_si128(a1 + 6);
            v25[7] = _mm_loadu_si128(a1 + 7);
            v25[8] = _mm_loadu_si128(a1 + 8);
            v25[9] = _mm_loadu_si128(a1 + 9);
            v25[10] = _mm_loadu_si128(a1 + 10);
            v25[11] = _mm_loadu_si128(a1 + 11);
            v25[12] = _mm_loadu_si128(a1 + 12);
          }
          else
          {
            if ( v24 != 6 )
              goto LABEL_101;
            sub_72BAF0(v48, a1[12].m128i_i64[0], unk_4F06A60);
          }
          if ( (*(_BYTE *)(v28 + 96) & 2) == 0
            && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v28 + 112) + 8LL) + 16LL) + 96LL) & 2) == 0
            || sub_8D6D50(a1) )
          {
            sub_620DE0(v50, *(_QWORD *)(v28 + 104));
            goto LABEL_37;
          }
          if ( a4[10].m128i_i8[13] != 6 )
          {
            i = v44;
            v12 = a1;
            *a10 = 1;
            break;
          }
          v36 = *(_QWORD *)(*(_QWORD *)(v41 + 16) + 104LL);
          if ( !v43 )
            goto LABEL_80;
          v37 = *(_QWORD *)(v43 + 104);
          if ( v36 >= v37 )
          {
            v36 -= v37;
LABEL_80:
            sub_620DE0(v50, v36);
LABEL_37:
            v26 = sub_620E90((__int64)v48);
            sub_621270((unsigned __int16 *)&v48[11], v50, v26, &v46);
            goto LABEL_38;
          }
          sub_620DE0(v50, v37 - v36);
          v38 = sub_620E90((__int64)v48);
          sub_6215F0((unsigned __int16 *)&v48[11], v50, v38, &v46);
LABEL_38:
          v27 = a4[10].m128i_i8[13];
          v25 = v48;
          if ( v27 == 1 )
          {
            *a4 = _mm_loadu_si128(v48);
            a4[1] = _mm_loadu_si128(v25 + 1);
            a4[2] = _mm_loadu_si128(v25 + 2);
            a4[3] = _mm_loadu_si128(v25 + 3);
            a4[4] = _mm_loadu_si128(v25 + 4);
            a4[5] = _mm_loadu_si128(v25 + 5);
            a4[6] = _mm_loadu_si128(v25 + 6);
            a4[7] = _mm_loadu_si128(v25 + 7);
            a4[8] = _mm_loadu_si128(v25 + 8);
            a4[9] = _mm_loadu_si128(v25 + 9);
            a4[10] = _mm_loadu_si128(v25 + 10);
            a4[11] = _mm_loadu_si128(v25 + 11);
            a4[12] = _mm_loadu_si128(v25 + 12);
          }
          else
          {
            if ( v27 != 6 )
LABEL_101:
              sub_721090(v25);
            a4[12].m128i_i64[0] = sub_620FA0((__int64)v48, &v46);
          }
          v20 = (_QWORD *)*v20;
          if ( (_QWORD *)*v42 == v20 )
            goto LABEL_46;
        }
      }
    }
    v29 = 0;
    if ( (*(_BYTE *)(a3 + 140) & 0xFB) == 8 )
      v29 = (unsigned int)sub_8D4C10(a3, dword_4F077C4 != 2);
    while ( *(_BYTE *)(i + 140) == 12 )
      i = *(_QWORD *)(i + 160);
    v30 = sub_73C570(i, v29, -1);
    for ( j = a4[8].m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v32 = *(_BYTE *)(j + 168);
    if ( (v32 & 1) != 0 )
    {
      if ( (v32 & 2) != 0 )
        v33 = sub_72D6A0(v30);
      else
        v33 = sub_72D600(v30);
    }
    else
    {
      v33 = sub_72D2E0(v30, 0);
    }
    a4[8].m128i_i64[0] = v33;
    v34 = a4[10].m128i_i8[8];
    a4[10].m128i_i8[8] = v34 | 8;
    if ( !a7 )
      a4[10].m128i_i8[8] = v34 | 0x28;
    if ( *a10 )
    {
      v49 = 0;
    }
    else if ( !a9 )
    {
      if ( v49 )
        goto LABEL_90;
      if ( v12[10].m128i_i8[13] == 6 && v12[11].m128i_i8[0] == 1 && (v12[10].m128i_i8[8] & 8) == 0 )
      {
        v49 = (_QWORD *)sub_731250(v12[11].m128i_i64[1]);
        if ( (unsigned int)sub_8D3410(*v49) )
          v49 = (_QWORD *)sub_6EE5A0((__int64)v49);
        if ( v49 )
        {
LABEL_90:
          sub_6E7420(a2, (_DWORD *)a3, 0, 0, 0, a7, 0, (__int64 *)&v49, a11, &v47);
          a4[9].m128i_i64[0] = (__int64)v49;
        }
      }
    }
    return (_DWORD *)sub_724E30(&v48);
  }
}
