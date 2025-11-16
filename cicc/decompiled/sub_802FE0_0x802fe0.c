// Function: sub_802FE0
// Address: 0x802fe0
//
void __fastcall sub_802FE0(__m128i *a1, __m128i *a2, int a3, __int64 *a4, int a5, __m128i *a6, _DWORD *a7, int a8)
{
  __int64 v11; // rdi
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdi
  int v15; // r9d
  __int16 v16; // r15
  int v17; // eax
  const __m128i *v18; // rdi
  __int64 v19; // rbx
  __m128i *v20; // rax
  __int64 v21; // r13
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rbx
  int v25; // [rsp+0h] [rbp-50h]
  int v26; // [rsp+4h] [rbp-4Ch]
  int v27; // [rsp+4h] [rbp-4Ch]
  int v28; // [rsp+4h] [rbp-4Ch]
  int v29; // [rsp+8h] [rbp-48h]
  int v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+14h] [rbp-3Ch] BYREF
  __m128i v32[3]; // [rsp+18h] [rbp-38h] BYREF

  v32[0].m128i_i64[0] = 0;
  v11 = a1[11].m128i_i64[0];
  v12 = dword_4D03EB8[0];
  if ( dword_4D03EB8[0] )
  {
    v12 = 0;
    if ( (*(_BYTE *)(v11 - 8) & 1) != 0 && (*(_BYTE *)(v11 + 50) & 1) != 0 )
    {
      v27 = a5;
      v30 = a3;
      v20 = sub_740B80(v11, 0xAu);
      a5 = v27;
      a3 = v30;
      v11 = (__int64)v20;
      v12 = 1;
    }
  }
  if ( a3 )
  {
    v13 = *(_QWORD *)(v11 + 8);
    v14 = *(_QWORD *)(v11 + 16);
    v15 = dword_4F07508[0];
    v16 = dword_4F07508[1];
    if ( v13 )
      *(_QWORD *)dword_4F07508 = *(_QWORD *)(v13 + 64);
    v26 = v15;
    v29 = v12;
    sub_7FE6E0(v14, (__int64)a2, 1, 0, a6);
    v17 = v29;
    LOWORD(dword_4F07508[1]) = v16;
    v18 = (const __m128i *)v32[0].m128i_i64[0];
    dword_4F07508[0] = v26;
    if ( v32[0].m128i_i64[0] )
      goto LABEL_7;
  }
  else
  {
    v28 = v12;
    v25 = a5;
    sub_802F60(v11, a1[8].m128i_i64[0], a6);
    sub_7FEC50(v11, a2, a4, 0, a8, v25, a6, 0, v32);
    v18 = (const __m128i *)v32[0].m128i_i64[0];
    v17 = v28;
    if ( v32[0].m128i_i64[0] )
    {
LABEL_7:
      v19 = a1[7].m128i_i64[1];
      if ( v17 )
      {
        v31 = 0;
        sub_7296C0(&v31);
        sub_740190(v32[0].m128i_i64[0], a1, 0);
        sub_729730(v31);
      }
      else
      {
        sub_72A510(v18, a1);
      }
      a1[7].m128i_i64[1] = v19;
      *a7 = 1;
      return;
    }
  }
  v21 = sub_7F9140((__int64)a2);
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_7E1F40(v21) )
    {
      v21 = (__int64)sub_72BA30(byte_4D03F80[0]);
    }
    else if ( (unsigned int)sub_7E1F90(v21) )
    {
      v21 = sub_7E1D00(v21, a2);
    }
  }
  else if ( (unsigned int)sub_8D2B20(v21) )
  {
    while ( *(_BYTE *)(v21 + 140) == 12 )
      v21 = *(_QWORD *)(v21 + 160);
    v21 = (__int64)sub_72C610(*(_BYTE *)(v21 + 160));
  }
  if ( (unsigned int)sub_8D3B80(v21) || (unsigned int)sub_8D2B80(v21) || (unsigned int)sub_8D2B50(v21) )
  {
    sub_724A80((__int64)a1, 10);
  }
  else
  {
    v22 = *(_BYTE *)(v21 + 140);
    if ( v22 == 12 )
    {
      v23 = v21;
      do
      {
        v23 = *(_QWORD *)(v23 + 160);
        v22 = *(_BYTE *)(v23 + 140);
      }
      while ( v22 == 12 );
    }
    if ( v22 == 20 )
    {
      a1[10].m128i_i8[13] = 15;
      a1[11].m128i_i8[0] = 0;
      a1[11].m128i_i64[1] = 0;
    }
    else
    {
      v24 = a1[7].m128i_i64[1];
      sub_72BB40(v21, a1);
      a1[7].m128i_i64[1] = v24;
    }
  }
}
