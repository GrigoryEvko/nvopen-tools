// Function: sub_7D6E80
// Address: 0x7d6e80
//
__int64 __fastcall sub_7D6E80(__m128i *a1, unsigned __int8 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // r15d
  int v6; // r14d
  unsigned int v9; // ebx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 result; // rax
  char v17; // cl
  unsigned __int8 v18; // dl
  _BYTE *v19; // rdi
  char v20; // si
  char v21; // r15
  __int64 *v22; // rdi
  __int64 *i; // rax
  __int64 v24; // r15
  char v25; // cl
  __int64 v26; // r14
  __m128i v27; // xmm7
  __int64 v28; // rax
  int v29; // ecx
  __m128i v30; // xmm3
  __int64 v31; // rax
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // [rsp+0h] [rbp-50h]
  unsigned __int8 v35; // [rsp+Fh] [rbp-41h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a4;
  v6 = a3;
  v9 = (_DWORD)a4 == 0 ? 2 : 16386;
  v10 = sub_7D5DD0(a1, v9, a3, a4, a5);
  v36[0] = v10;
  if ( !v10 )
    goto LABEL_36;
  v11 = *(unsigned __int8 *)(v10 + 80);
  if ( (_BYTE)v11 != 19 )
  {
    if ( !v5 || dword_4D047B4 )
      goto LABEL_5;
    goto LABEL_24;
  }
  if ( unk_4F04C48 != -1 )
  {
    v12 = (__int64)qword_4F04C68;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
    {
      sub_85F9C0(v36);
      if ( v36[0] )
      {
        if ( *(_BYTE *)(v36[0] + 80) != 19 )
          goto LABEL_5;
        goto LABEL_27;
      }
LABEL_36:
      if ( v5 )
        return v36[0];
      goto LABEL_29;
    }
  }
  if ( !v5 )
    goto LABEL_27;
  v12 = dword_4D047B4;
  if ( dword_4D047B4 )
    goto LABEL_27;
LABEL_24:
  v15 = a1[1].m128i_i64[1];
  if ( !v15 )
  {
    if ( (_BYTE)v11 != 19 )
      goto LABEL_8;
    goto LABEL_27;
  }
  if ( *(_BYTE *)(v15 + 80) == 24 )
  {
    v36[0] = 0;
    if ( (a1[1].m128i_i8[1] & 0x40) == 0 )
    {
      a1[1].m128i_i8[0] &= ~0x80u;
      a1[1].m128i_i64[1] = 0;
    }
    return v36[0];
  }
  if ( (_BYTE)v11 == 19 )
  {
LABEL_27:
    v36[0] = 0;
    if ( (a1[1].m128i_i8[1] & 0x40) == 0 )
    {
      a1[1].m128i_i8[0] &= ~0x80u;
      a1[1].m128i_i64[1] = 0;
    }
LABEL_29:
    result = sub_7D5DD0(a1, v9 | 0x20, v11, v12, v13);
    v36[0] = result;
    if ( !result )
      return result;
    if ( *(_BYTE *)(result + 80) == 19 )
    {
      result = 0;
      if ( (a1[1].m128i_i8[1] & 0x40) == 0 )
      {
        a1[1].m128i_i8[0] &= ~0x80u;
        a1[1].m128i_i64[1] = 0;
      }
      return result;
    }
LABEL_5:
    v15 = a1[1].m128i_i64[1];
    if ( !v15 )
      goto LABEL_8;
  }
  if ( (*(_BYTE *)(v15 + 82) & 4) != 0 )
    sub_87DC80(a1, 0, 0, 1, v13, v14);
LABEL_8:
  result = v36[0];
  v17 = *(_BYTE *)(v36[0] + 81);
  v18 = *(_BYTE *)(v36[0] + 80);
  v19 = (_BYTE *)v36[0];
  v20 = v17 & 0x40;
  if ( v18 == 3 )
  {
    if ( *(_BYTE *)(v36[0] + 104) )
    {
      v21 = 1;
      if ( !v20 )
        goto LABEL_10;
    }
    else
    {
      if ( !v20 )
      {
        if ( !dword_4F077BC || qword_4F077A8 > 0x76BFu )
          goto LABEL_74;
        v22 = *(__int64 **)(v36[0] + 88);
        for ( i = v22; *((_BYTE *)i + 140) == 12; i = (__int64 *)i[20] )
          ;
        v24 = *i;
        if ( v6 || (*(_BYTE *)(v36[0] + 81) & 0x10) != 0 )
        {
          if ( (unsigned int)sub_8D3A70(v22) )
            goto LABEL_48;
          v19 = (_BYTE *)v36[0];
          v33 = *(_BYTE *)(v36[0] + 80);
          if ( v33 == 6 )
            goto LABEL_47;
          if ( v33 != 3 )
          {
LABEL_73:
            v17 = v19[81];
LABEL_74:
            if ( v17 >= 0 )
            {
              sub_6851A0(0x4B1u, dword_4F07508, *(_QWORD *)(*(_QWORD *)v19 + 8LL));
              goto LABEL_59;
            }
            v18 = v19[80];
            result = (__int64)v19;
            v25 = v18 != 3;
            v21 = 0;
            goto LABEL_51;
          }
          v22 = *(__int64 **)(v36[0] + 88);
        }
        if ( !(unsigned int)sub_8D2870(v22) )
        {
LABEL_72:
          v19 = (_BYTE *)v36[0];
          if ( !v36[0] )
            return 0;
          goto LABEL_73;
        }
LABEL_47:
        if ( (unsigned int)sub_879510(v24) )
        {
LABEL_48:
          v21 = 0;
          goto LABEL_49;
        }
        goto LABEL_72;
      }
      v21 = 0;
    }
LABEL_54:
    v26 = sub_8D2250(*(_QWORD *)(v36[0] + 88));
    v34 = *(_QWORD *)v26;
    v35 = byte_4F07472[0];
    sub_6849F0(byte_4F07472[0], 0x31Au, dword_4F07508, *(_QWORD *)(*(_QWORD *)v36[0] + 8LL));
    result = v34;
    if ( v35 == 8 )
    {
      *a1 = _mm_loadu_si128(xmmword_4F06660);
      a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v30 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v31 = *(_QWORD *)dword_4F07508;
      a1[1].m128i_i8[1] |= 0x20u;
      a1->m128i_i64[1] = v31;
      a1[3] = v30;
      return 0;
    }
    if ( *(_BYTE *)(v26 + 140) != 14 )
    {
      if ( !v34 || *(_BYTE *)(v34 + 80) != a2 )
      {
        sub_686470(0x22Bu, dword_4F07508, (__int64)*(&off_4AF8080 + a2), v26);
LABEL_59:
        *a1 = _mm_loadu_si128(xmmword_4F06660);
        a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
        a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
        v27 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v28 = *(_QWORD *)dword_4F07508;
        a1[1].m128i_i8[1] |= 0x20u;
        a1->m128i_i64[1] = v28;
        a1[3] = v27;
        return 0;
      }
      v36[0] = v34;
      v18 = a2;
      v25 = v21 | (a2 != 3);
      goto LABEL_51;
    }
LABEL_49:
    result = v36[0];
    if ( !v36[0] )
      return result;
    v18 = *(_BYTE *)(v36[0] + 80);
    v25 = v21 | (v18 != 3);
LABEL_51:
    if ( v25 )
      goto LABEL_10;
    return v36[0];
  }
  v21 = 0;
  if ( v20 )
    goto LABEL_54;
LABEL_10:
  if ( v18 == a2 )
    goto LABEL_15;
  if ( dword_4F077C4 == 2 || unk_4F07778 > 199900 || dword_4F077C0 )
  {
    if ( !qword_4D0495C || v21 )
    {
LABEL_15:
      if ( a1[1].m128i_i8[0] < 0 )
        sub_685440(unk_4F07470, 0x196u, a1[1].m128i_i64[1]);
      if ( dword_4F077C4 == 2 )
      {
        v32 = a1[1].m128i_i64[1];
        if ( v32 )
        {
          if ( (*(_DWORD *)(v32 + 80) & 0x41000) != 0 )
          {
            sub_8841F0(a1, 0, 0, 0);
            return v36[0];
          }
        }
      }
      return v36[0];
    }
  }
  else if ( v21 )
  {
    goto LABEL_15;
  }
  v29 = *(_DWORD *)(result + 40);
  result = 0;
  if ( v29 == *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C) )
    goto LABEL_15;
  return result;
}
