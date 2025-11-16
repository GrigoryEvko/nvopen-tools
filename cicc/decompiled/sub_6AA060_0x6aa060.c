// Function: sub_6AA060
// Address: 0x6aa060
//
__int64 __fastcall sub_6AA060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r15d
  int v11; // eax
  __int64 v13; // rax
  char i; // dl
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // rdx
  __int16 v30; // [rsp+10h] [rbp-260h]
  char *v31; // [rsp+10h] [rbp-260h]
  __int64 v32; // [rsp+10h] [rbp-260h]
  int v33; // [rsp+1Ch] [rbp-254h]
  __int64 v34; // [rsp+28h] [rbp-248h] BYREF
  __int64 v35; // [rsp+30h] [rbp-240h] BYREF
  __int64 v36; // [rsp+38h] [rbp-238h] BYREF
  _BYTE v37[160]; // [rsp+40h] [rbp-230h] BYREF
  _QWORD v38[2]; // [rsp+E0h] [rbp-190h] BYREF
  char v39; // [rsp+F0h] [rbp-180h]
  __int64 v40; // [rsp+168h] [rbp-108h]

  v36 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(qword_4D03C50 + 40LL);
  sub_6E1E00(5, v37, 0, 0);
  *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x20u;
  ++*(_BYTE *)(qword_4F061C8 + 75LL);
  v33 = sub_679C10(0x405u);
  if ( v33 )
  {
    sub_65CD60(&v34);
    v5 = (_QWORD *)sub_726700(22);
    v33 = 0;
    *v5 = sub_72CBE0(22, v37, v6, v7, v8, v9);
    v5[7] = v34;
    *(_QWORD *)((char *)v5 + 28) = *(_QWORD *)&dword_4F063F8;
  }
  else
  {
    sub_69ED20((__int64)v38, 0, 0, 1);
    if ( !v39 )
      goto LABEL_11;
    v13 = v38[0];
    for ( i = *(_BYTE *)(v38[0] + 140LL); i == 12; i = *(_BYTE *)(v13 + 140) )
      v13 = *(_QWORD *)(v13 + 160);
    if ( !i )
    {
LABEL_11:
      sub_7BE280(67, 253, 0, 0);
      --*(_BYTE *)(qword_4F061C8 + 75LL);
      if ( sub_5CBA20(word_4F06418[0]) )
        goto LABEL_5;
      goto LABEL_4;
    }
    sub_6F69D0(v38, 8);
    if ( v39 == 3 )
    {
      v28 = *(_BYTE *)(v40 + 80);
      if ( v28 == 17 )
      {
        v33 = 1;
        v5 = (_QWORD *)sub_731280(*(_QWORD *)(*(_QWORD *)(v40 + 88) + 88LL));
      }
      else
      {
        if ( v28 != 20 || (v29 = *(_QWORD *)(*(_QWORD *)(v40 + 88) + 104LL), ((*(_BYTE *)(v29 + 120) - 2) & 0xFD) != 0) )
          sub_721090(v38);
        v33 = 1;
        v5 = (_QWORD *)sub_731280(*(_QWORD *)(v29 + 192));
      }
    }
    else
    {
      v5 = (_QWORD *)sub_6F6F40(v38, 0);
    }
  }
  sub_7BE280(67, 253, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  if ( !sub_5CBA20(word_4F06418[0]) )
  {
LABEL_4:
    sub_6E5C80(8, 3052, &dword_4F063F8);
LABEL_5:
    v10 = qword_4F063F0;
    v30 = WORD2(qword_4F063F0);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_6E6260(a1);
    goto LABEL_6;
  }
  v31 = *(char **)(qword_4D04A00 + 8);
  v15 = 1;
  if ( !sub_5D1850(v31, 0, 1) )
  {
    v15 = 2;
    if ( !sub_5D1850(v31, 0, 2) )
    {
      sub_6E5DE0(8, 1097, &dword_4F063F8, v31);
      goto LABEL_5;
    }
  }
  v32 = sub_5CBA50(v15, 0);
  if ( !v32 )
  {
    sub_6E6000(v15, 0, v16, v17, v18, v19);
    goto LABEL_5;
  }
  v20 = sub_724DC0(v15, 0, v16, v17, v18, v19);
  v35 = v20;
  v21 = sub_72C390();
  sub_72BB40(v21, v20);
  v22 = sub_740630(v35);
  *(_QWORD *)(v22 + 104) = v32;
  *(_BYTE *)(v32 + 10) = 21;
  v23 = sub_726700(2);
  v5[2] = v23;
  *(_QWORD *)(v23 + 56) = v22;
  *(_QWORD *)v5[2] = *(_QWORD *)(v22 + 128);
  sub_5CB870(v32);
  sub_724E30(&v35);
  v10 = qword_4F063F0;
  v30 = WORD2(qword_4F063F0);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  v24 = sub_726700(23);
  v25 = sub_72C390();
  *(_QWORD *)(v24 + 64) = v5;
  *(_QWORD *)v24 = v25;
  v26 = v36;
  *(_BYTE *)(v24 + 56) = 70;
  *(_QWORD *)(v24 + 28) = v26;
  sub_6E3AC0(v24, &v36, 0, 0);
  sub_6E2E50(2, a1);
  sub_7197C0(v24, a1 + 144, *(_BYTE *)(qword_4D03C50 + 16LL) != 0, &v36, &v35);
  v27 = *(_QWORD *)(a1 + 272);
  *(_BYTE *)(a1 + 17) = 2;
  *(_QWORD *)a1 = v27;
  if ( v33 )
    sub_620D80((_WORD *)(a1 + 320), 0);
LABEL_6:
  v11 = v36;
  *(_DWORD *)(a1 + 76) = v10;
  *(_DWORD *)(a1 + 68) = v11;
  *(_WORD *)(a1 + 72) = WORD2(v36);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 68);
  *(_WORD *)(a1 + 80) = v30;
  unk_4F061D8 = *(_QWORD *)(a1 + 76);
  sub_6E3280(a1, &v36);
  return sub_6E2B30(a1, &v36);
}
