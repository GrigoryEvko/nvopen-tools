// Function: sub_38454A0
// Address: 0x38454a0
//
unsigned __int8 *__fastcall sub_38454A0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  int *v6; // r12
  char v7; // si
  __int64 v8; // rcx
  int v9; // edi
  int v10; // r10d
  __int64 v11; // r11
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // r12
  __int64 v15; // rdx
  unsigned __int16 v16; // ax
  __int64 v17; // r13
  __int16 *v18; // rdx
  __int16 v19; // ax
  __int64 v20; // rdx
  int v21; // esi
  __int64 v22; // rax
  unsigned __int16 v23; // ax
  __int64 v24; // r9
  int v25; // r9d
  unsigned __int8 *v26; // r12
  __int64 v28; // rax
  int v29; // eax
  int v30; // r12d
  __int128 v31; // [rsp-20h] [rbp-80h]
  unsigned __int16 v32; // [rsp+Eh] [rbp-52h]
  __int64 v33; // [rsp+10h] [rbp-50h] BYREF
  int v34; // [rsp+18h] [rbp-48h]
  int v35; // [rsp+20h] [rbp-40h] BYREF
  __int64 v36; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v33 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v33, v5, 1);
  v34 = *(_DWORD *)(a2 + 72);
  v35 = sub_375D5B0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = sub_3805BC0(a1 + 712, &v35);
  sub_37593F0(a1, v6);
  v7 = *(_BYTE *)(a1 + 512) & 1;
  if ( v7 )
  {
    v8 = a1 + 520;
    v9 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 528);
    v8 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v22 )
      goto LABEL_20;
    v9 = v22 - 1;
  }
  v10 = *v6;
  LODWORD(v11) = v9 & (37 * *v6);
  v12 = v8 + 24LL * (unsigned int)v11;
  v13 = *(_DWORD *)v12;
  if ( *v6 == *(_DWORD *)v12 )
    goto LABEL_6;
  v29 = 1;
  while ( v13 != -1 )
  {
    v30 = v29 + 1;
    v11 = v9 & (unsigned int)(v29 + v11);
    v12 = v8 + 24 * v11;
    v13 = *(_DWORD *)v12;
    if ( v10 == *(_DWORD *)v12 )
      goto LABEL_6;
    v29 = v30;
  }
  if ( v7 )
  {
    v28 = 192;
    goto LABEL_21;
  }
  v22 = *(unsigned int *)(a1 + 528);
LABEL_20:
  v28 = 24 * v22;
LABEL_21:
  v12 = v8 + v28;
LABEL_6:
  v14 = *(_QWORD *)(v12 + 8);
  v15 = *(unsigned int *)(v12 + 16);
  v16 = *(_WORD *)(*(_QWORD *)(v14 + 48) + 16 * v15);
  v17 = v15;
  v18 = *(__int16 **)(a2 + 48);
  v32 = v16;
  v19 = *v18;
  v20 = *((_QWORD *)v18 + 1);
  LOWORD(v35) = v19;
  v36 = v20;
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 176) > 0x34u )
    {
LABEL_8:
      v21 = word_4456340[(unsigned __int16)v35 - 1];
      goto LABEL_13;
    }
  }
  else if ( !sub_3007100((__int64)&v35) )
  {
    goto LABEL_12;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v35 )
  {
    if ( (unsigned __int16)(v35 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_8;
  }
LABEL_12:
  v21 = sub_3007130((__int64)&v35, 0xFFFFFFFF00000000LL);
LABEL_13:
  v23 = sub_2D43050(word_4456580[v32 - 1], v21);
  *((_QWORD *)&v31 + 1) = v17;
  *(_QWORD *)&v31 = v14;
  sub_3406EB0(*(_QWORD **)(a1 + 8), 0xA1u, (__int64)&v33, v23, 0, v24, v31, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  v26 = sub_33FAF80(
          *(_QWORD *)(a1 + 8),
          216,
          (__int64)&v33,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v25,
          a3);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v26;
}
