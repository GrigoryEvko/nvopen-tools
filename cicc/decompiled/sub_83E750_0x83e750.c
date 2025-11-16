// Function: sub_83E750
// Address: 0x83e750
//
void __fastcall sub_83E750(const __m128i *a1, __int64 *a2, FILE *a3)
{
  __int8 v4; // cl
  const __m128i *v5; // r12
  const __m128i *v6; // rax
  __int8 v7; // dl
  __int64 *v8; // rax
  __int64 v9; // r14
  __int64 v10; // r14
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int *v14; // rax
  __int64 v15; // rsi
  FILE *v16; // rdx
  __int64 v17; // rdi
  char v18; // al
  int v19; // [rsp+0h] [rbp-30h] BYREF
  int v20; // [rsp+4h] [rbp-2Ch] BYREF
  int v21; // [rsp+8h] [rbp-28h] BYREF
  int v22; // [rsp+Ch] [rbp-24h] BYREF

  v4 = a1[8].m128i_i8[12];
  if ( v4 == 12 )
  {
    v5 = a1;
    do
      v5 = (const __m128i *)v5[10].m128i_i64[0];
    while ( v5[8].m128i_i8[12] == 12 );
    v19 = 0;
    v6 = a1;
    v20 = 0;
    v21 = 0;
    do
    {
      v6 = (const __m128i *)v6[10].m128i_i64[0];
      v7 = v6[8].m128i_i8[12];
    }
    while ( v7 == 12 );
  }
  else
  {
    v19 = 0;
    v5 = a1;
    v7 = v4;
    v20 = 0;
    v21 = 0;
  }
  if ( v7 )
  {
    if ( dword_4D04460 )
      goto LABEL_20;
    if ( !a2 )
    {
      v15 = 0;
      if ( (v4 & 0xFB) == 8 )
        v15 = (unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2);
      v10 = sub_83DE00(v5, v15, 1, 0, (__int64)a3, &v19, &v20, 0, &v21);
      if ( v21 )
      {
        sub_6E61E0((__int64)v5, (__int64)a3, 1u);
        goto LABEL_20;
      }
      goto LABEL_11;
    }
    v8 = *(__int64 **)(a2[19] + 168);
    v9 = *v8;
    if ( !(unsigned int)sub_8D3070(*(_QWORD *)(*v8 + 8)) )
    {
LABEL_10:
      v10 = *a2;
LABEL_11:
      if ( v19 )
      {
        sub_6E5D20(unk_4F07471, 0x122u, a3, (__int64)v5);
      }
      else
      {
        if ( v20 )
        {
          if ( sub_6E53E0(unk_4F07471, 0x2B4u, a3) )
            sub_6853B0(unk_4F07471, 0x2B4u, a3, v10);
          goto LABEL_20;
        }
        if ( v10 )
        {
          if ( !(unsigned int)sub_6E6010() || (unsigned int)sub_884000(v10, 1) )
          {
            sub_732910(*(_QWORD *)(v10 + 88), (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x42) == 2, 1, v11, v12, v13);
            v16 = 0;
            if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
              v16 = a3;
            if ( (unsigned int)sub_875C60(v10, 1, v16) || *(char *)(qword_4D03C50 + 18LL) >= 0 )
              goto LABEL_20;
          }
          else
          {
            v22 = 0;
            v14 = 0;
            if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
              v14 = &v22;
            sub_87D9B0(v10, 0, 0, (_DWORD)a3, 0, unk_4F07471, 691, (__int64)v14);
            if ( !v22 )
              goto LABEL_20;
          }
          sub_6E50A0();
        }
        else
        {
          sub_6E5D20(unk_4F07471, 0x14Eu, a3, (__int64)v5);
        }
      }
LABEL_20:
      sub_82AFD0((__int64)v5, (__int64)a3);
      return;
    }
    v17 = sub_8D46C0(*(_QWORD *)(v9 + 8));
    if ( (*(_BYTE *)(v17 + 140) & 0xFB) == 8 )
    {
      v18 = sub_8D4C10(v17, dword_4F077C4 != 2);
      if ( dword_4F077BC || !dword_4D04964 )
      {
LABEL_40:
        if ( (v18 & 1) != 0 )
          goto LABEL_10;
        goto LABEL_37;
      }
      if ( dword_4F077C4 != 2 )
        goto LABEL_48;
      if ( unk_4F07778 > 201102 )
        goto LABEL_40;
    }
    else
    {
      if ( dword_4F077BC || !dword_4D04964 || dword_4F077C4 != 2 || unk_4F07778 > 201102 )
        goto LABEL_37;
      v18 = 0;
    }
    if ( dword_4F07774 )
      goto LABEL_40;
LABEL_48:
    if ( (v18 & 3) != 3 )
      goto LABEL_40;
LABEL_37:
    v20 = 1;
    goto LABEL_10;
  }
}
