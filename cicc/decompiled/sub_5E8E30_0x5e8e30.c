// Function: sub_5E8E30
// Address: 0x5e8e30
//
void __fastcall sub_5E8E30(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // rbx
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdi
  char v11; // al
  char v12; // al
  __int64 v13; // r15
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 i; // r12
  char v18; // al
  __int64 v19; // rsi
  __int64 v21; // [rsp+18h] [rbp-98h]
  char v23; // [rsp+3Eh] [rbp-72h] BYREF
  char v24; // [rsp+3Fh] [rbp-71h] BYREF
  _QWORD v25[2]; // [rsp+40h] [rbp-70h] BYREF
  __m128i v26; // [rsp+50h] [rbp-60h]
  __m128i v27; // [rsp+60h] [rbp-50h]
  __m128i v28; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(a2 + 80) == 17 )
  {
    for ( i = *(_QWORD *)(a2 + 88); i; i = *(_QWORD *)(i + 8) )
    {
      if ( *a3 )
        break;
      v18 = *(_BYTE *)(i + 80);
      v19 = i;
      if ( v18 == 16 )
      {
        v19 = **(_QWORD **)(i + 88);
        v18 = *(_BYTE *)(v19 + 80);
      }
      if ( v18 == 24 )
        v19 = *(_QWORD *)(v19 + 88);
      sub_5E8E30(a1, v19, a3);
    }
    if ( v3 )
      return;
LABEL_34:
    sub_6851C0(1001, dword_4F07508);
    *a3 = 1;
    return;
  }
  if ( !v3 )
    goto LABEL_34;
  while ( 1 )
  {
    v4 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v5 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v25[1] = unk_4D04A08;
    v26 = v4;
    v27 = v5;
    v28 = v6;
    v7 = *(_QWORD *)(v3 + 40);
    v25[0] = unk_4D04A00;
    v8 = sub_7D2AC0(v25, v7, 0);
    v9 = v8;
    if ( v8 )
    {
      if ( a2 == v8 )
        return;
      if ( *(_BYTE *)(v8 + 80) == 17 )
      {
        v10 = a2;
        v11 = *(_BYTE *)(a2 + 80);
        if ( v11 == 16 )
        {
          v10 = **(_QWORD **)(a2 + 88);
          v11 = *(_BYTE *)(v10 + 80);
        }
        if ( v11 == 24 )
          v10 = *(_QWORD *)(v10 + 88);
        v21 = sub_87D510(v10, &v23);
        v12 = *(_BYTE *)(v9 + 80);
        if ( v12 == 16 )
        {
          v9 = **(_QWORD **)(v9 + 88);
          v12 = *(_BYTE *)(v9 + 80);
        }
        if ( v12 == 24 )
          v9 = *(_QWORD *)(v9 + 88);
        v13 = *(_QWORD *)(v9 + 88);
        if ( v13 )
          break;
      }
    }
LABEL_4:
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      goto LABEL_34;
  }
  while ( 1 )
  {
    v14 = *(_BYTE *)(v13 + 80);
    v15 = v13;
    if ( v14 == 16 )
    {
      v15 = **(_QWORD **)(v13 + 88);
      v14 = *(_BYTE *)(v15 + 80);
    }
    if ( v14 == 24 )
      v15 = *(_QWORD *)(v15 + 88);
    v16 = sub_87D510(v15, &v24);
    if ( v24 == v23 )
    {
      if ( (unsigned int)sub_8C7EB0(v16, v21) )
        break;
    }
    v13 = *(_QWORD *)(v13 + 8);
    if ( !v13 )
      goto LABEL_4;
  }
}
