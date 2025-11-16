// Function: sub_6AFBA0
// Address: 0x6afba0
//
__int64 __fastcall sub_6AFBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rsi
  __int64 v7; // rdi
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // eax
  __int64 v16; // rax
  bool v17; // zf
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 i; // rdi
  __int64 v26; // r13
  __int16 v27; // [rsp+12h] [rbp-31Eh]
  int v28; // [rsp+14h] [rbp-31Ch]
  int v29; // [rsp+24h] [rbp-30Ch] BYREF
  __int64 v30; // [rsp+28h] [rbp-308h] BYREF
  __int64 v31; // [rsp+30h] [rbp-300h] BYREF
  __int64 v32; // [rsp+38h] [rbp-2F8h] BYREF
  _QWORD v33[44]; // [rsp+40h] [rbp-2F0h] BYREF
  _BYTE v34[68]; // [rsp+1A0h] [rbp-190h] BYREF
  __int64 v35; // [rsp+1E4h] [rbp-14Ch]
  __int64 v36; // [rsp+1ECh] [rbp-144h]
  __int64 v37; // [rsp+230h] [rbp-100h]

  if ( a1 )
  {
    sub_6F8AB0(a1, (unsigned int)v33, (unsigned int)v34, 0, (unsigned int)&v31, (unsigned int)&v29, 0);
    v30 = *(_QWORD *)(v37 + 56);
    v32 = v35;
  }
  else
  {
    v31 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    v23 = qword_4F061C8;
    v24 = qword_4D03C50;
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(v24 + 40);
    ++*(_BYTE *)(v23 + 75);
    sub_69ED20((__int64)v33, 0, 0, 1);
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    sub_7BE280(67, 253, 0, 0);
    v32 = *(_QWORD *)&dword_4F063F8;
    sub_65CD60(&v30);
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  v6 = 0;
  sub_6F69D0(v33, 0);
  if ( !(unsigned int)sub_8D2B80(v30) )
  {
    v7 = v30;
    v8 = *(_BYTE *)(v30 + 140);
    if ( v8 == 12 )
    {
      v9 = v30;
      do
      {
        v9 = *(_QWORD *)(v9 + 160);
        v8 = *(_BYTE *)(v9 + 140);
      }
      while ( v8 == 12 );
    }
    if ( v8 )
    {
      if ( dword_4F04C44 != -1
        || (v22 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v22 + 6) & 6) != 0)
        || *(_BYTE *)(v22 + 4) == 12 )
      {
        if ( (unsigned int)sub_8DBE70(v30) )
          goto LABEL_8;
        v7 = v30;
      }
      v6 = &v32;
      sub_685360(0xAE9u, &v32, v7);
      v30 = sub_72C930(2793);
    }
  }
LABEL_8:
  v28 = qword_4F063F0;
  v27 = WORD2(qword_4F063F0);
  v10 = sub_726700(22);
  *(_QWORD *)v10 = sub_72CBE0(22, v6, v11, v12, v13, v14);
  *(_QWORD *)(v10 + 28) = v32;
  v15 = sub_688510(a1, v33, &v29);
  if ( !v29 )
  {
    if ( !v15 )
    {
      v16 = sub_72C930(a1);
      v17 = *(_BYTE *)(v10 + 24) == 22;
      v30 = v16;
      *(_QWORD *)(v10 + 56) = v16;
      if ( !v17 )
        goto LABEL_15;
      goto LABEL_11;
    }
    if ( (unsigned int)sub_8D2B80(v30) )
    {
      for ( i = v30; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( !(unsigned int)sub_8DBF30() )
      {
        v26 = sub_8D4620(v33[0]);
        if ( v26 != sub_8D4620(v30) )
          sub_6E5ED0(2794, &v31, v33[0], v30);
      }
    }
  }
  v17 = *(_BYTE *)(v10 + 24) == 22;
  *(_QWORD *)(v10 + 56) = v30;
  if ( !v17 )
    goto LABEL_15;
LABEL_11:
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    sub_6E70E0(v10, v34);
    LODWORD(v35) = v31;
    WORD2(v35) = WORD2(v31);
    *(_QWORD *)dword_4F07508 = v35;
    v36 = *(_QWORD *)&dword_4F077C8;
    unk_4F061D8 = *(_QWORD *)&dword_4F077C8;
    sub_6E3280(v34, &dword_4F077C8);
    sub_6F6F40(v34, 0);
  }
LABEL_15:
  v18 = sub_726700(23);
  v19 = v30;
  *(_BYTE *)(v18 + 56) = 59;
  *(_QWORD *)v18 = v19;
  v20 = sub_6F6F40(v33, 0);
  *(_QWORD *)(v20 + 16) = v10;
  *(_QWORD *)(v18 + 64) = v20;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && *(_BYTE *)(v18 + 24) )
  {
    sub_6E70E0(v18, v34);
    LODWORD(v35) = v31;
    WORD2(v35) = WORD2(v31);
    *(_QWORD *)dword_4F07508 = v35;
    LODWORD(v36) = v28;
    WORD2(v36) = v27;
    unk_4F061D8 = v36;
    sub_6E3280(v34, &dword_4F077C8);
    sub_6F6F40(v34, 0);
  }
  sub_6E70E0(v18, a2);
  *(_DWORD *)(a2 + 68) = v31;
  *(_WORD *)(a2 + 72) = WORD2(v31);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v28;
  *(_WORD *)(a2 + 80) = v27;
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  return sub_6E3280(a2, &v31);
}
