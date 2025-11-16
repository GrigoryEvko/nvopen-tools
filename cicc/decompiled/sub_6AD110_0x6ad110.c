// Function: sub_6AD110
// Address: 0x6ad110
//
__int64 __fastcall sub_6AD110(int a1, __int64 *a2, _DWORD *a3, __int64 *a4, _DWORD *a5, _QWORD *a6, __m128i *a7)
{
  __int64 *v8; // r12
  _QWORD *v10; // rdx
  char v11; // si
  __int64 v12; // rsi
  int v13; // r10d
  bool v14; // r13
  __int64 v15; // rax
  unsigned int v16; // r13d
  const __m128i *v17; // rax
  unsigned int v18; // r12d
  __int64 v19; // rcx
  unsigned int v21; // esi
  unsigned int v22; // esi
  __int64 v23; // rcx
  __int64 v24; // r8
  char v25; // al
  unsigned int v26; // [rsp+Ch] [rbp-1C4h]
  __m128i *v28; // [rsp+18h] [rbp-1B8h]
  char v29; // [rsp+20h] [rbp-1B0h]
  int v30; // [rsp+20h] [rbp-1B0h]
  char v31; // [rsp+26h] [rbp-1AAh]
  char v32; // [rsp+27h] [rbp-1A9h]
  int v34; // [rsp+3Ch] [rbp-194h] BYREF
  _BYTE v35[400]; // [rsp+40h] [rbp-190h] BYREF

  v8 = a4;
  v10 = (_QWORD *)qword_4D03C50;
  v11 = *(_BYTE *)(qword_4D03C50 + 20LL) >> 3;
  *(_BYTE *)(qword_4D03C50 + 20LL) &= ~8u;
  v12 = v11 & 1;
  v32 = v12;
  if ( (_DWORD)v12 )
  {
    if ( unk_4F04C48 != -1 )
    {
      v10 = qword_4F04C68;
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
      {
        v29 = 1;
        v31 = v12;
        if ( !dword_4F077BC )
          goto LABEL_5;
        goto LABEL_17;
      }
    }
    v25 = 0;
    if ( dword_4F04C44 != -1 )
      v25 = v12;
    v31 = v25;
    v29 = dword_4F04C44 != -1;
  }
  else
  {
    v31 = 0;
    v29 = 0;
  }
  if ( !dword_4F077BC )
    goto LABEL_5;
LABEL_17:
  if ( qword_4F077A8 <= 0x9DCFu || (a4 = (__int64 *)v35, v13 = 18, v28 = (__m128i *)v35, a1 != 5) )
  {
LABEL_5:
    v13 = 2;
    v28 = 0;
    v14 = a1 == 3;
    if ( a2 )
      goto LABEL_6;
LABEL_20:
    v26 = v13;
    *(_QWORD *)a3 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(a3, v12, v10, a4);
    sub_7BE280(43, 438, 0, 0);
    ++*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
    ++*(_BYTE *)(qword_4F061C8 + 52LL);
    *(_QWORD *)a5 = *(_QWORD *)&dword_4F063F8;
    *v8 = sub_65CDF0(*(_BYTE *)(qword_4D03C50 + 16LL) <= 3u, 1, &v34, 0, 0);
    if ( dword_4D04324 )
    {
      if ( a1 == 7 )
        v21 = 878;
      else
        v21 = 879;
      sub_684AB0(a3, v21);
    }
    v16 = dword_4D0478C != 0 && v14;
    v30 = sub_68E4D0(v8, (__int64)a5, v34, v16, dword_4D0478C, v29);
    sub_7BE280(44, 439, 0, 0);
    --*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 632);
    --*(_BYTE *)(qword_4F061C8 + 52LL);
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_69ED20((__int64)a7, v28, 0, v26);
LABEL_8:
    if ( !v31 )
      goto LABEL_9;
    goto LABEL_26;
  }
  v14 = 0;
  if ( !a2 )
    goto LABEL_20;
LABEL_6:
  sub_6FE5D0((_DWORD)a2, 0, (_DWORD)a3, (_DWORD)v8, (_DWORD)a5, 0, 0, (__int64)a7, (__int64)v28);
  v15 = *a2;
  v34 = 0;
  *a6 = *(_QWORD *)(v15 + 44);
  if ( !dword_4D04324 )
  {
    v16 = dword_4D0478C != 0 && v14;
    v30 = sub_68E4D0(v8, (__int64)a5, 0, v16, dword_4D0478C, v29);
    goto LABEL_8;
  }
  v22 = 878;
  if ( a1 != 7 )
    v22 = 879;
  sub_684AB0(a3, v22);
  v16 = dword_4D0478C != 0 && v14;
  v30 = sub_68E4D0(v8, (__int64)a5, v34, v16, dword_4D0478C, v29);
  if ( !v31 )
  {
LABEL_9:
    v17 = a7;
    if ( (a7[1].m128i_i8[2] & 1) == 0 )
      goto LABEL_10;
LABEL_28:
    sub_68FA30(*v8, a3, v17, (__int64)v28);
    if ( !v16 )
      goto LABEL_11;
LABEL_29:
    if ( (unsigned int)sub_8D3410(*v8) )
    {
      if ( !dword_4D0478C )
      {
        v18 = sub_68BAB0(*v8, a7, a5, v23, v24);
        if ( !v18 )
          goto LABEL_12;
      }
    }
    goto LABEL_11;
  }
LABEL_26:
  if ( !(unsigned int)sub_8D23B0(*v8) )
    goto LABEL_9;
  sub_697260(*v8, (__int64)a7, (__int64)a5);
  v17 = a7;
  if ( (a7[1].m128i_i8[2] & 1) != 0 )
    goto LABEL_28;
LABEL_10:
  if ( v16 )
    goto LABEL_29;
LABEL_11:
  v18 = v30 == 0;
LABEL_12:
  if ( a2 )
  {
    v19 = qword_4D03C50;
  }
  else
  {
    *a6 = qword_4F063F0;
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    v19 = qword_4D03C50;
    --*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  *(_BYTE *)(v19 + 20) = (8 * (v32 & 1)) | *(_BYTE *)(v19 + 20) & 0xF7;
  return v18;
}
