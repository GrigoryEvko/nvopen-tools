// Function: sub_7BF130
// Address: 0x7bf130
//
__int64 __fastcall sub_7BF130(unsigned int a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // r12
  char v5; // al
  unsigned __int16 v6; // ax
  int v7; // eax
  bool v8; // zf
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rax
  int v13; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v14; // [rsp+8h] [rbp-38h] BYREF

  v13 = 0;
  if ( !(unsigned int)sub_7C8410(a1 & 0xFFFFFFE3, a2, a3) )
  {
    v4 = sub_7D5DD0(&qword_4D04A00, dword_3C19800[(int)a2]);
    goto LABEL_18;
  }
  v4 = (__int64)qword_4D04A18;
  if ( !qword_4D04A18 )
    goto LABEL_11;
  v5 = *((_BYTE *)qword_4D04A18 + 80);
  if ( v5 == 16 )
  {
    v4 = *(_QWORD *)qword_4D04A18[11];
    v5 = *(_BYTE *)(v4 + 80);
  }
  if ( v5 == 24 )
  {
    v4 = *(_QWORD *)(v4 + 88);
LABEL_18:
    if ( v4 )
    {
      v5 = *(_BYTE *)(v4 + 80);
      if ( v5 == 19 )
        goto LABEL_20;
      goto LABEL_7;
    }
LABEL_11:
    v4 = 0;
    goto LABEL_12;
  }
  if ( v5 == 19 )
    goto LABEL_20;
LABEL_7:
  if ( v5 != 3 )
  {
    if ( v5 != 21 )
      goto LABEL_12;
    if ( (unk_4D04A12 & 4) == 0 )
    {
      v6 = sub_7BE840(0, 0);
      v4 = sub_7C7F70(v4, dword_4F06650[0], a1, v6, &v13);
      goto LABEL_12;
    }
    goto LABEL_22;
  }
  if ( !*(_BYTE *)(v4 + 104) )
    goto LABEL_12;
  v12 = *(_QWORD *)(v4 + 88);
  if ( (*(_BYTE *)(v12 + 177) & 0x10) == 0 || !*(_QWORD *)(*(_QWORD *)(v12 + 168) + 168LL) )
    goto LABEL_12;
LABEL_20:
  if ( (unk_4D04A12 & 4) == 0 )
  {
    v4 = sub_7BF840(v4, a1, &v13);
    goto LABEL_12;
  }
LABEL_22:
  if ( (unsigned int)a2 > 0xA )
  {
    if ( (a1 & 0x4000000) != 0 )
    {
LABEL_24:
      v14 = xmmword_4D04A20.m128i_i64[1];
      qword_4D04A18 = (_QWORD *)sub_8A0370(v4, (unsigned int)&v14, 0, 0, 0, 0, 0);
      v4 = (__int64)qword_4D04A18;
      v11 = *(_DWORD *)&word_4D04A10 & 0xFFFBBFFF;
      BYTE1(v11) = ((unsigned __int16)(word_4D04A10 & 0xBFFF) >> 8) | 0x40;
      *(_DWORD *)&word_4D04A10 = v11;
      qword_4D04A00 = *qword_4D04A18;
    }
  }
  else
  {
    v10 = 1234;
    if ( _bittest64(&v10, (unsigned int)a2)
      || (a1 & 0x4000000) != 0
      || (_DWORD)a2 == 2 && dword_4F04C64 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 2) != 0 )
    {
      goto LABEL_24;
    }
  }
LABEL_12:
  v7 = v13 | *a3;
  v8 = v13 == 0;
  *a3 = v7;
  if ( !v8 )
    v4 = 0;
  if ( (a1 & 0x1C) != 0 )
    v7 = *a3 | sub_7AC1A0(a1, dword_4F07508);
  *a3 = v7;
  return v4;
}
