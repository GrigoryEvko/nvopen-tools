// Function: sub_877120
// Address: 0x877120
//
__int64 __fastcall sub_877120(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rsi
  char v4; // dl
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  const char *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = (_QWORD *)unk_4D04A48;
  if ( !unk_4D04A48 )
  {
LABEL_13:
    v6 = (_QWORD *)sub_823970(24);
    *v6 = 0;
    v1 = v6;
    v6[1] = 0;
    v7 = unk_4D04A48;
    v1[2] = a1;
    *v1 = v7;
    unk_4D04A48 = v1;
    v8 = sub_877070();
    v1[1] = v8;
    v9 = v8;
    v10 = (const char *)sub_67D020(a1, v14);
    v11 = v14[0] + 10LL;
    *(_QWORD *)(v9 + 16) = v14[0] + 9LL;
    v12 = sub_7279A0(v11);
    *(_QWORD *)(v9 + 8) = v12;
    *(_QWORD *)v12 = 0x726F74617265706FLL;
    *(_BYTE *)(v12 + 8) = 32;
    strcpy((char *)(*(_QWORD *)(v9 + 8) + 9LL), v10);
    return v1[1];
  }
  v2 = 0;
  while ( 1 )
  {
    v3 = v1[2];
    if ( v3 == a1 )
      break;
    if ( (unsigned int)sub_8DED30(a1, v3, 0x40000) )
      goto LABEL_11;
LABEL_4:
    v2 = v1;
    if ( !*v1 )
      goto LABEL_13;
    v1 = (_QWORD *)*v1;
  }
  v4 = *(_BYTE *)(a1 + 140);
  if ( v4 == 12 )
  {
    v5 = a1;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v4 = *(_BYTE *)(v5 + 140);
    }
    while ( v4 == 12 );
  }
  if ( !v4 )
    goto LABEL_4;
LABEL_11:
  if ( v2 )
  {
    *v2 = *v1;
    *v1 = unk_4D04A48;
    unk_4D04A48 = v1;
  }
  return v1[1];
}
