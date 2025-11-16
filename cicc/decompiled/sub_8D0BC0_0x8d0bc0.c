// Function: sub_8D0BC0
// Address: 0x8d0bc0
//
void __fastcall sub_8D0BC0(char *s, int a2, _QWORD *a3)
{
  _QWORD *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rcx
  _QWORD *v14; // rax
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9

  if ( qword_4D03FF0 )
    sub_8D06C0((_QWORD *)qword_4D03FF0);
  dword_4F063F8 = 0;
  word_4F063FC[0] = 0;
  dword_4D03FE8[0] = a2;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  unk_4D03FD8 = a3 != 0;
  qword_4D03FE0 = s;
  sub_727950();
  if ( dword_4D03FE8[0] )
    sub_705EE0();
  v5 = (_QWORD *)sub_823970(416);
  v6 = qword_4F60530;
  *v5 = 0;
  v7 = (__int64)v5;
  v8 = sub_823970(v6);
  *(_QWORD *)(v7 + 8) = 0;
  *(_QWORD *)(v7 + 16) = v8;
  sub_85FFE0(v7 + 24);
  *(_QWORD *)(v7 + 184) = 0;
  v12 = 256;
  *(_QWORD *)(v7 + 344) = 0;
  *(_QWORD *)(v7 + 176) = 0;
  memset(
    (void *)((v7 + 192) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v7 - (((_DWORD)v7 + 192) & 0xFFFFFFF8) + 352) >> 3));
  v13 = 0;
  *(_WORD *)(v7 + 384) = 256;
  *(_QWORD *)(v7 + 352) = 0;
  v14 = (_QWORD *)qword_4F60540;
  *(_QWORD *)(v7 + 360) = 0;
  *(_QWORD *)(v7 + 368) = 0;
  *(_DWORD *)(v7 + 400) = 0;
  for ( *(_QWORD *)(v7 + 408) = 0; v14; v14 = (_QWORD *)*v14 )
  {
    v9 = v14[4];
    if ( v9 )
    {
      v13 = v14[1];
      *(_QWORD *)(v7 + v9) = v13;
    }
  }
  *(_QWORD *)(v7 + 368) = a3;
  *(_BYTE *)(v7 + 384) = a3 == 0;
  if ( !qword_4D03FD0 )
    qword_4D03FD0 = (_QWORD *)v7;
  v15 = v7;
  qword_4D03FF0 = v7;
  sub_8D0A80((_QWORD *)v7, 256, v9, v13, v10, v11);
  if ( qword_4F60528 )
    *(_QWORD *)qword_4F60528 = v7;
  qword_4F60528 = v7;
  if ( a3 )
  {
    v16 = a3[7];
    a3[2] = v7;
    qword_4D04750 = v16;
    v17 = a3[6];
    qword_4F076A8 = a3[4];
    unk_4F07688 = v17;
    qword_4F076A0 = a3[5];
    v12 = (__int64)&qword_4F076A8;
    qword_4F076E8 = (unsigned __int8 *)sub_722430(s, 1);
    sub_7209D0((__int64)qword_4F076E8, &qword_4F076A8, &qword_4F076A0);
    sub_706250();
    *(_DWORD *)(v7 + 400) = dword_4F073B8[0];
    v15 = a3[3];
    sub_721790(v15);
    if ( !dword_4D04944 )
      goto LABEL_15;
LABEL_21:
    sub_7061A0();
    sub_858C60(v15, (unsigned int *)v12, v24, v25, v26, v27);
    goto LABEL_17;
  }
  sub_706250();
  *(_DWORD *)(v7 + 400) = dword_4F073B8[0];
  if ( dword_4D04944 )
    goto LABEL_21;
LABEL_15:
  if ( unk_4D04508 && !dword_4D03CAC )
  {
    sub_706190();
    sub_852E40(v15, (unsigned int *)v12, v20, v21, v22, v23);
  }
  sub_7061A0();
  sub_666B30(v15, v12, v18, v19);
LABEL_17:
  sub_709290(v15, (_DWORD *)v12);
  sub_8D0B10();
}
