// Function: sub_8DC7D0
// Address: 0x8dc7d0
//
__m128i *__fastcall sub_8DC7D0(__int64 a1)
{
  const __m128i *v1; // r12
  __int64 v2; // rbx
  bool v3; // r14
  __int8 v4; // al
  int v5; // r13d
  char v6; // bl
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __m128i *v13; // rbx
  __int64 v14; // rax
  __m128i *v15; // rbx

  v1 = (const __m128i *)a1;
  LOBYTE(v2) = dword_4F60558;
  if ( dword_4F60558 )
  {
    LOBYTE(v2) = 0;
    goto LABEL_3;
  }
  if ( *(_BYTE *)(a1 + 140) != 12 || !dword_4F07590 )
  {
LABEL_3:
    if ( dword_4F04C44 == -1 )
    {
      v3 = 0;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
LABEL_6:
        v4 = *(_BYTE *)(a1 + 140);
        if ( v4 != 12 )
        {
LABEL_34:
          if ( v4 == 8 && (v1[10].m128i_i8[9] & 4) != 0 )
          {
            v13 = (__m128i *)sub_7259C0(8);
            sub_73C230(v1, v13);
            v1 = v13;
            v13[10].m128i_i8[9] &= ~4u;
          }
          return sub_8DC200((__int64)v1, (unsigned int (__fastcall *)(__m128i *, _QWORD, __m128i **))sub_8DCAB0, 0);
        }
        goto LABEL_7;
      }
    }
LABEL_5:
    v3 = (unsigned int)sub_866AA0() != 0;
    goto LABEL_6;
  }
  if ( dword_4F04C44 != -1
    || (v14 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v14 + 6) & 6) != 0)
    || *(_BYTE *)(v14 + 4) == 12 )
  {
    v2 = (unsigned int)sub_8DBE70(a1) == 0;
    goto LABEL_3;
  }
  if ( (*(_BYTE *)(v14 + 6) & 2) != 0 )
    goto LABEL_5;
  v3 = 0;
LABEL_7:
  v5 = 0;
  v6 = v2 & 1;
  while ( (v1[11].m128i_i8[10] & 8) == 0 || !(unsigned int)sub_8DBE70(v1[10].m128i_i64[0]) )
  {
    if ( !v3 && (v1[5].m128i_i8[9] & 1) != 0 )
      goto LABEL_11;
    v8 = v6 | (dword_4F07590 == 0);
    if ( dword_4F60558 )
    {
      if ( v8 )
        goto LABEL_9;
    }
    else if ( v8 )
    {
      if ( (v1[11].m128i_i8[10] & 0x20) != 0 )
        goto LABEL_11;
LABEL_9:
      if ( (v1[5].m128i_i8[9] & 4) != 0 )
      {
        v7 = **(_QWORD **)(v1[2].m128i_i64[1] + 32);
        if ( (unsigned __int8)(*(_BYTE *)(v7 + 80) - 4) > 1u )
          goto LABEL_11;
        v9 = *(_QWORD *)(v7 + 88);
        if ( !v9 || (*(_BYTE *)(v9 + 177) & 0x20) != 0 )
          goto LABEL_11;
      }
    }
    v10 = v1[11].m128i_u8[8];
    if ( (unsigned __int8)v10 > 0xCu
      || (v11 = 6338, !_bittest64(&v11, v10))
      || *(_QWORD *)(v1[10].m128i_i64[1] + 24) && !v5
      || (_BYTE)v10 == 7
      || (unsigned int)sub_8D96C0(v1[10].m128i_i64[0]) )
    {
      v4 = v1[8].m128i_i8[12];
      if ( v4 != 12 )
        goto LABEL_34;
      goto LABEL_28;
    }
    v5 = 1;
LABEL_11:
    v1 = (const __m128i *)v1[10].m128i_i64[0];
    v4 = v1[8].m128i_i8[12];
    if ( v4 != 12 )
      goto LABEL_34;
  }
  v4 = v1[8].m128i_i8[12];
  if ( v4 != 12 )
    goto LABEL_34;
LABEL_28:
  if ( (v1[11].m128i_i8[9] & 0x70) != 0 )
  {
    if ( (v1[11].m128i_i8[9] & 0xF) != 0 )
    {
      v15 = (__m128i *)sub_7259C0(12);
      sub_73C230(v1, v15);
      v1 = v15;
      v15[11].m128i_i8[9] &= 0xF0u;
    }
    else
    {
      v1 = (const __m128i *)v1[10].m128i_i64[0];
    }
  }
  return sub_8DC200((__int64)v1, (unsigned int (__fastcall *)(__m128i *, _QWORD, __m128i **))sub_8DCAB0, 0);
}
