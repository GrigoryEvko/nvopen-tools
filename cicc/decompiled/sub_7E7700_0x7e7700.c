// Function: sub_7E7700
// Address: 0x7e7700
//
void __fastcall sub_7E7700(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  _QWORD *v6; // rax
  int v7; // edi
  __int64 v8; // rax
  __int64 *v9; // rbx
  _QWORD *v10; // rax
  unsigned __int8 v11; // al
  __m128i *v12; // rax
  __m128i *v13; // rsi
  char **v14; // rax
  const char *v15; // rdi
  char **v16; // r14
  size_t v17; // rax
  char *v18; // rax
  char v19; // al
  __m128i *v20; // rax
  int v21; // edi
  char v22; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v23[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (*(_BYTE *)(a1 + 176) & 0x40) == 0 || *(_BYTE *)(*(_QWORD *)(a1 + 120) + 140LL) != 14 )
  {
    if ( unk_4D045AC )
    {
      v14 = (char **)sub_7247C0(24);
      v15 = *(const char **)(a1 + 8);
      v16 = v14;
      if ( v15 )
      {
        v17 = strlen(v15);
        v18 = (char *)sub_7247C0(v17 + 1);
        *v16 = strcpy(v18, *(const char **)(a1 + 8));
      }
      else
      {
        *v14 = (char *)byte_3F871B3;
      }
      v16[1] = (char *)a1;
      v16[2] = *(char **)(a2 + 128);
      *(_QWORD *)(a2 + 128) = v16;
    }
    sub_813140(a1, 7, 0, a3, a2);
  }
  *(_BYTE *)(a1 + 89) &= ~1u;
  sub_72B850(a1);
  if ( *(_QWORD *)(a1 + 8)
    && ((*(_BYTE *)(a1 + 176) & 0x40) == 0 || *(_BYTE *)(*(_QWORD *)(a1 + 120) + 140LL) != 14)
    && (unsigned int)sub_736A50(a3) )
  {
    v19 = *(_BYTE *)(a1 + 88);
    *(_BYTE *)(a1 + 156) |= 0x80u;
    *(_BYTE *)(a1 + 136) = 0;
    *(_BYTE *)(a1 + 88) = v19 & 0x8F | 0x30;
    sub_7E4C10(a1);
    *(_BYTE *)(a1 + 168) = *(_BYTE *)(a3 + 200) & 7 | *(_BYTE *)(a1 + 168) & 0xF8;
  }
  sub_735E40(a1, 0);
  v5 = *(_BYTE *)(a1 + 177);
  *(_BYTE *)(a1 + 173) |= 0x80u;
  if ( v5 != 4 )
  {
    if ( v5 != 1 )
      goto LABEL_12;
    if ( *(_BYTE *)(a1 + 136) || *(_QWORD *)(a1 + 240) )
      goto LABEL_14;
    goto LABEL_40;
  }
  v9 = *(__int64 **)(a2 + 200);
  if ( a1 == v9[1] )
  {
    *(_QWORD *)(a2 + 200) = *v9;
  }
  else
  {
    do
    {
      v10 = v9;
      v9 = (__int64 *)*v9;
    }
    while ( v9[1] != a1 );
    *v10 = *v9;
  }
  *v9 = unk_4D03F40;
  unk_4D03F40 = v9;
  *(_BYTE *)(a1 + 173) |= 0x40u;
  v11 = *((_BYTE *)v9 + 16);
  *(_BYTE *)(a1 + 177) = v11;
  if ( v11 == 2 )
  {
    *(_QWORD *)(a1 + 184) = v9[3];
    goto LABEL_14;
  }
  if ( v11 <= 2u )
  {
    if ( !v11 )
      goto LABEL_14;
    LODWORD(v23[0]) = 0;
    if ( (unsigned int)sub_7E76A0(v9[3]) )
    {
      v12 = sub_740190(v9[3], 0, 0x40u);
      *(_QWORD *)(a1 + 184) = v12;
      v13 = v12;
LABEL_30:
      sub_7FACF0(a1, v13);
LABEL_31:
      v5 = *(_BYTE *)(a1 + 177);
LABEL_12:
      if ( v5 == 5 )
      {
        sub_7296C0(v23);
        v6 = sub_73B8B0(*(const __m128i **)(a1 + 184), 512);
        v7 = v23[0];
        *(_QWORD *)(a1 + 184) = v6;
        sub_729730(v7);
      }
      goto LABEL_14;
    }
    sub_7296C0(v23);
    v20 = sub_740190(v9[3], 0, 0x40u);
    v21 = v23[0];
    *(_QWORD *)(a1 + 184) = v20;
    sub_729730(v21);
    if ( *(_BYTE *)(a1 + 136) || *(_QWORD *)(a1 + 240) )
      goto LABEL_31;
LABEL_40:
    v13 = *(__m128i **)(a1 + 184);
    goto LABEL_30;
  }
  if ( v11 != 3 )
    sub_721090();
LABEL_14:
  if ( qword_4F04C50 )
  {
    v8 = *(_QWORD *)(qword_4F04C50 + 32LL);
    if ( v8 )
    {
      if ( (*(_BYTE *)(v8 + 198) & 0x10) != 0 )
      {
        sub_72F9F0(a1, 0, &v22, v23);
        sub_7E1230((_BYTE *)a1, 0, 0, 0);
        if ( v22 == 1 )
          sub_7E1270(*(_QWORD *)v23[0]);
      }
    }
  }
}
