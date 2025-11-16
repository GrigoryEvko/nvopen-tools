// Function: sub_6B0100
// Address: 0x6b0100
//
__int64 __fastcall sub_6B0100(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  int v5; // ebx
  __int64 v6; // rcx
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 i; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned int v22; // [rsp+4h] [rbp-48Ch]
  __int64 v23; // [rsp+8h] [rbp-488h]
  __int64 v24; // [rsp+8h] [rbp-488h]
  __int16 v25; // [rsp+1Ah] [rbp-476h]
  int v26; // [rsp+1Ch] [rbp-474h]
  char v27; // [rsp+2Ch] [rbp-464h] BYREF
  int v28; // [rsp+30h] [rbp-460h] BYREF
  unsigned int v29; // [rsp+34h] [rbp-45Ch] BYREF
  __int64 v30; // [rsp+38h] [rbp-458h] BYREF
  __int64 v31[44]; // [rsp+40h] [rbp-450h] BYREF
  __int64 v32[44]; // [rsp+1A0h] [rbp-2F0h] BYREF
  _QWORD v33[8]; // [rsp+300h] [rbp-190h] BYREF
  __int64 v34; // [rsp+344h] [rbp-14Ch]
  __int64 v35; // [rsp+34Ch] [rbp-144h]

  v4 = a1;
  if ( a1 )
  {
    v25 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
    v26 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
    sub_6F8AB0(a1, (unsigned int)v31, (unsigned int)v32, 0, (unsigned int)&v30, (unsigned int)&v27, 0);
  }
  else
  {
    v30 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    v16 = qword_4D03C50;
    v17 = qword_4F061C8;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(v16 + 40);
    ++*(_BYTE *)(v17 + 75);
    sub_69ED20((__int64)v31, 0, 0, 1);
    sub_7BE280(67, 253, 0, 0);
    sub_69ED20((__int64)v32, 0, 0, 1);
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    v26 = qword_4F063F0;
    v25 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  v5 = sub_688E10(a1, v31, &v28);
  if ( v5 )
  {
    v22 = sub_688E10(a1, v32, &v29);
    if ( !v22 )
    {
LABEL_6:
      if ( v29 )
        goto LABEL_14;
      goto LABEL_7;
    }
    v8 = v31[0];
    if ( v31[0] == v32[0] || (unsigned int)sub_8DED30(v31[0], v32[0], 1) )
      goto LABEL_14;
    if ( v4 )
    {
      *(_BYTE *)(v4 + 56) = 1;
    }
    else
    {
      v8 = 2531;
      sub_6861A0(0x9E3u, &v30, v31[0], v32[0]);
    }
    v6 = sub_72C930(v8);
  }
  else
  {
    if ( v28 )
    {
      v22 = sub_688E10(a1, v32, &v29);
      if ( v22 )
        goto LABEL_14;
      goto LABEL_6;
    }
    v24 = sub_72C930(a1);
    v18 = sub_688E10(a1, v32, &v29);
    v6 = v24;
    v22 = v18;
    if ( !v18 )
    {
      a1 = v29;
      if ( !v29 )
      {
LABEL_7:
        v22 = 0;
        v6 = sub_72C930(a1);
      }
    }
  }
  if ( v6 )
  {
    sub_6E6260(a2);
    goto LABEL_10;
  }
LABEL_14:
  if ( v28 )
  {
    sub_6F40C0(v31);
    v23 = *(_QWORD *)&dword_4D03B80;
  }
  else
  {
    sub_6F69D0(v31, 0);
    v23 = 0;
  }
  v9 = sub_6F6F40(v31, 0);
  if ( v29 )
  {
    sub_6F40C0(v32);
    v23 = *(_QWORD *)&dword_4D03B80;
  }
  else
  {
    sub_6F69D0(v32, 0);
  }
  v10 = sub_6F6F40(v32, 0);
  *(_QWORD *)(v9 + 16) = v10;
  v11 = v10;
  if ( !v23 )
  {
    for ( i = v31[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v23 = sub_72C6F0(*(unsigned __int8 *)(i + 160));
  }
  v15 = sub_726700(23);
  *(_BYTE *)(v15 + 56) = 54;
  *(_QWORD *)v15 = v23;
  *(_QWORD *)(v15 + 64) = v9;
  if ( v5 && *(_BYTE *)(v9 + 24) == 2 && v22 && *(_BYTE *)(v11 + 24) == 2 )
  {
    v33[0] = sub_724DC0(23, 0, v22, v12, v13, v14);
    sub_724C70(v33[0], 4);
    v20 = v33[0];
    *(_QWORD *)(v33[0] + 128LL) = v23;
    *(__m128i *)*(_QWORD *)(v20 + 176) = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v9 + 56) + 176LL));
    v21 = v33[0];
    *(__m128i *)(*(_QWORD *)(v33[0] + 176LL) + 16LL) = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v11 + 56) + 176LL));
    *(_QWORD *)(v21 + 144) = v15;
    sub_6E6A50(v21, a2);
    sub_724E30(v33);
  }
  else
  {
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v15 + 24) )
    {
      sub_6E70E0(v15, v33);
      LODWORD(v34) = v30;
      WORD2(v34) = WORD2(v30);
      *(_QWORD *)dword_4F07508 = v34;
      LODWORD(v35) = v26;
      WORD2(v35) = v25;
      unk_4F061D8 = v35;
      sub_6E3280(v33, &dword_4F077C8);
      sub_6F6F40(v33, 0);
    }
    sub_6E70E0(v15, a2);
  }
LABEL_10:
  *(_DWORD *)(a2 + 68) = v30;
  *(_WORD *)(a2 + 72) = WORD2(v30);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v26;
  *(_WORD *)(a2 + 80) = v25;
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  return sub_6E3280(a2, &v30);
}
