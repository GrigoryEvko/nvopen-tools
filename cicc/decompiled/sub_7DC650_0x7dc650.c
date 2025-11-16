// Function: sub_7DC650
// Address: 0x7dc650
//
__int64 __fastcall sub_7DC650(__int64 a1)
{
  __int64 v1; // r15
  unsigned int v4; // eax
  const __m128i *v5; // rax
  __m128i *v6; // r12
  int v7; // ebx
  const char *v8; // r15
  size_t v9; // rax
  char *v10; // r13
  const char *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r8
  __m128i *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  bool v17; // sf
  const char *v18; // r13
  size_t v19; // rax
  char *v20; // rax
  char *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned int v25; // [rsp+4h] [rbp-3Ch]
  unsigned int v26; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 152);
  if ( !v1 )
  {
    v4 = sub_7DB6D0(a1);
    v5 = (const __m128i *)sub_7DB910(v4, a1);
    v6 = sub_73C570(v5, 1);
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
    {
      v18 = (const char *)sub_80FAD0(a1);
      v19 = strlen(v18);
      v20 = (char *)sub_7E1510(v19 + 1);
      v21 = strcpy(v20, v18);
      v15 = 1;
      v22 = sub_7E2190(v21);
      *(_BYTE *)(v22 + 89) |= 8u;
      v17 = *(char *)(a1 - 8) < 0;
      v1 = v22;
      *(_QWORD *)(a1 + 152) = v22;
      if ( v17 )
      {
        v15 = 7;
        sub_75B260(v22, 7u);
      }
    }
    else
    {
      v7 = unk_4D04358;
      if ( !unk_4D04358
        && ((unsigned int)sub_8D2600(a1)
         || (unsigned int)sub_8D2780(a1)
         || (unsigned int)sub_8D2AC0(a1)
         || (unsigned int)sub_7E1E50(a1)
         || (unsigned int)sub_8D2E30(a1)
         && ((v23 = sub_8D46C0(a1), (*(_BYTE *)(v23 + 140) & 0xFB) != 8)
          || (unsigned int)sub_8D4C10(v23, dword_4F077C4 != 2) <= 1)
         && ((unsigned int)sub_8D2600(v23) || (unsigned int)sub_8D2AC0(v23) || (unsigned int)sub_8D2780(v23))) )
      {
        v26 = 0;
        v25 = 0;
      }
      else
      {
        v26 = sub_7DB020(a1);
        if ( v26 )
        {
          v26 = 0;
          v7 = 1;
          v25 = 1;
        }
        else
        {
          v25 = sub_8D96C0(a1);
          if ( v25 )
          {
            v25 = 0;
            v7 = 1;
            if ( (*(_BYTE *)(a1 + 89) & 1) != 0 )
            {
              v24 = sub_72B7F0(a1);
              v26 = sub_736A50(v24) != 0;
            }
          }
          else
          {
            v26 = 1;
            v7 = 1;
          }
        }
      }
      v8 = (const char *)sub_80FAD0(a1);
      v9 = strlen(v8);
      v10 = (char *)sub_7E1510(v9 + 1);
      strcpy(v10, v8);
      v1 = *(_QWORD *)(unk_4F07288 + 112LL);
      if ( v1 )
      {
        while ( 1 )
        {
          v11 = *(const char **)(v1 + 8);
          if ( v11 )
          {
            if ( *v11 == *v10 && !strcmp(v11, v10) )
              break;
          }
          v1 = *(_QWORD *)(v1 + 112);
          if ( !v1 )
            goto LABEL_15;
        }
        v14 = *(__m128i **)(v1 + 120);
        if ( v6 == v14 || (unsigned int)sub_8D97D0(v14, v6, 0, v12, v13) )
        {
          *(_QWORD *)(a1 + 152) = v1;
          return v1;
        }
      }
LABEL_15:
      v15 = 1;
      v16 = sub_7E2190(v10);
      *(_BYTE *)(v16 + 89) |= 8u;
      v17 = *(char *)(a1 - 8) < 0;
      v1 = v16;
      *(_QWORD *)(a1 + 152) = v16;
      if ( v17 )
      {
        v15 = 7;
        sub_75B260(v16, 7u);
      }
      if ( v7 )
      {
        sub_7DCA00(a1, v25, v26);
        return v1;
      }
    }
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u )
    {
      if ( *(_BYTE *)(v1 + 136) != 2 )
        *(_BYTE *)(v1 + 168) = sub_8DD330(a1, v15) & 7 | *(_BYTE *)(v1 + 168) & 0xF8;
      if ( dword_4F189C0 && *(_BYTE *)(*(_QWORD *)(a1 + 152) + 136LL) == 1 )
        sub_7DD730(a1, v15);
    }
  }
  return v1;
}
