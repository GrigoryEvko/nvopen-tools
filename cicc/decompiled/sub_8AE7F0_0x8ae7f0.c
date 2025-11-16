// Function: sub_8AE7F0
// Address: 0x8ae7f0
//
__m128i *__fastcall sub_8AE7F0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, int a5, __int64 *a6)
{
  char v7; // al
  char v8; // r15
  bool v9; // r14
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v20; // rcx
  __int64 v21; // rdi
  unsigned int *v22; // rax
  __m128i *v23; // rdi
  unsigned int *v24; // rsi
  __int64 v25; // rdi
  _BOOL4 v26; // eax
  __int64 v27; // r8
  __int64 *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __m128i *i; // rdi
  __m128i *j; // rdi
  __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-140h]
  __int64 v35; // [rsp+8h] [rbp-138h]
  unsigned int v37; // [rsp+24h] [rbp-11Ch]
  int v41; // [rsp+3Ch] [rbp-104h]
  unsigned __int16 v42; // [rsp+42h] [rbp-FEh]
  unsigned int v43; // [rsp+44h] [rbp-FCh]
  __int64 v44; // [rsp+48h] [rbp-F8h]
  const char *v45; // [rsp+50h] [rbp-F0h]
  __int16 v46; // [rsp+58h] [rbp-E8h]
  unsigned __int16 v47; // [rsp+5Ah] [rbp-E6h]
  int v48; // [rsp+5Ch] [rbp-E4h]
  __m128i *v49; // [rsp+68h] [rbp-D8h] BYREF
  _BYTE v50[64]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD v51[18]; // [rsp+B0h] [rbp-90h] BYREF

  v7 = *(_BYTE *)(a3 + 56);
  v8 = *(_BYTE *)(a3 + 72) & 1;
  v9 = (v7 & 2) != 0;
  v43 = dword_4F063F8;
  v42 = word_4F063FC[0];
  v45 = qword_4F06410;
  v44 = qword_4F06408;
  v48 = dword_4F07508[0];
  v46 = dword_4F07508[1];
  v41 = dword_4F061D8;
  v47 = word_4F061DC[0];
  if ( dword_4F04C44 != -1
    || (v33 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v33 + 6) & 6) != 0)
    || *(_BYTE *)(v33 + 4) == 12 )
  {
    if ( (v7 & 4) == 0 )
    {
LABEL_27:
      v37 = 2052;
      goto LABEL_4;
    }
LABEL_3:
    v37 = 2050;
    goto LABEL_4;
  }
  v37 = sub_89A370(a4);
  if ( v37 )
  {
    if ( (*(_BYTE *)(a3 + 56) & 4) == 0 )
      goto LABEL_27;
    goto LABEL_3;
  }
  if ( (*(_BYTE *)(a3 + 56) & 4) != 0 )
    goto LABEL_3;
LABEL_4:
  if ( !v8 )
  {
    v49 = *(__m128i **)(*(_QWORD *)(a2 + 88) + 128LL);
    goto LABEL_11;
  }
  if ( dword_4F6017C == unk_4D042F0 )
  {
    sub_6851C0(0x3FCu, dword_4F07508);
    v49 = (__m128i *)sub_72C930();
LABEL_11:
    if ( !a5 )
      goto LABEL_9;
    goto LABEL_12;
  }
  memset(v51, 0, 0x58u);
  v10 = *(_QWORD *)(a3 + 48);
  ++dword_4F6017C;
  sub_865840(v10, 0, 0, 0, a1, (__int64)a4, v37);
  sub_7BC160(a3 + 16);
  sub_8AE280((__int64)v50, &v49, 0, 0, 0, 0, 0, 0, v51);
  while ( word_4F06418[0] != 9 )
    sub_7B8B50((unsigned __int64)v50, (unsigned int *)&v49, v11, v12, v13, v14);
  sub_7B8B50((unsigned __int64)v50, (unsigned int *)&v49, v11, v12, v13, v14);
  sub_863FE0((__int64)v50, (__int64)&v49, v15, v16, v17, v18);
  --dword_4F6017C;
  if ( a5 )
  {
LABEL_12:
    if ( v9 || (*(_BYTE *)(a3 + 56) & 4) != 0 )
    {
      v20 = (unsigned int)dword_4F6017C;
      *(_BYTE *)(a3 + 56) &= ~4u;
      if ( v20 == unk_4D042F0 )
      {
        sub_6851C0(0x3FCu, dword_4F07508);
        *a6 = (__int64)sub_72C9A0();
      }
      else
      {
        v21 = *(_QWORD *)(a3 + 128);
        dword_4F6017C = v20 + 1;
        sub_865840(v21, 0, 0, 0, a1, (__int64)a4, v37);
        sub_7BC160(a3 + 96);
        v51[0] = *(_QWORD *)&dword_4F063F8;
        v22 = (unsigned int *)sub_724D80(0);
        v23 = v49;
        v24 = v22;
        *a6 = (__int64)v22;
        sub_679650((__int64)v23, v22);
        v25 = *a6;
        v26 = sub_72A990(*a6);
        v29 = v34;
        v30 = v35;
        if ( v26 )
        {
          v24 = (unsigned int *)v51;
          sub_6851C0(0x1DBu, v51);
          v25 = *a6;
          sub_72C970(*a6);
        }
        sub_863FE0(v25, (__int64)v24, v29, v30, v27, v28);
        --dword_4F6017C;
      }
    }
    else
    {
      *a6 = *(_QWORD *)(a3 + 80);
    }
    for ( i = v49; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
      ;
    if ( (unsigned int)sub_8D3F00(i) )
    {
      for ( j = v49; j[8].m128i_i8[12] == 12; j = (__m128i *)j[10].m128i_i64[0] )
        ;
      sub_65C2A0((__int64)j, *(_QWORD *)(*a6 + 128));
    }
  }
LABEL_9:
  dword_4F07508[0] = v48;
  LOWORD(dword_4F07508[1]) = v46;
  qword_4F06408 = v44;
  qword_4F06410 = v45;
  dword_4F063F8 = v43;
  word_4F063FC[0] = v42;
  dword_4F061D8 = v41;
  word_4F061DC[0] = v47;
  return v49;
}
