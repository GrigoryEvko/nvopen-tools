// Function: sub_7DA8C0
// Address: 0x7da8c0
//
_DWORD *__fastcall sub_7DA8C0(int a1)
{
  __int64 v1; // r15
  __int64 v2; // rbx
  int v3; // r14d
  char v4; // al
  _BYTE v6[144]; // [rsp+0h] [rbp-90h] BYREF

  v1 = qword_4F04C50;
  v2 = *(_QWORD *)(unk_4F072B0 + 8LL * a1);
  qword_4F04C50 = 0;
  v3 = dword_4F07270[0];
  dword_4D03F94 = 1;
  unk_4D03F68 = 0;
  qword_4F06BC0 = 0;
  sub_7296B0(a1);
  if ( *(_BYTE *)(v2 + 28) == 17 )
    sub_7E18E0(v6, unk_4F07288, 0);
  sub_7DA4E0(v2);
  v4 = *(_BYTE *)(v2 + 28);
  if ( !v4 )
  {
    sub_7DA790();
    if ( *(_BYTE *)(v2 + 28) != 17 )
      goto LABEL_5;
LABEL_7:
    sub_7E1AA0();
    goto LABEL_5;
  }
  if ( v4 == 17 )
    goto LABEL_7;
LABEL_5:
  qword_4F04C50 = v1;
  dword_4D03F94 = 0;
  return sub_7296B0(v3);
}
