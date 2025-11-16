// Function: sub_6DED10
// Address: 0x6ded10
//
__int64 __fastcall sub_6DED10(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx

  v2 = qword_4D03A98;
  if ( qword_4D03A98 )
    qword_4D03A98 = *(_QWORD *)(qword_4D03A98 + 40);
  else
    v2 = sub_823970(56);
  *(_QWORD *)v2 = 4;
  v3 = *(_BYTE *)(qword_4D03C50 + 17LL);
  *(_BYTE *)(v2 + 8) = 0;
  *(_QWORD *)(v2 + 16) = a1;
  *(_QWORD *)(v2 + 24) = 0;
  *(_QWORD *)v2 = (-(__int64)((v3 & 0x40) == 0) & 0xFFFFFFFFFFFFC000LL) + 16388;
  v4 = *a2;
  *(_QWORD *)(v2 + 40) = 0;
  *(_QWORD *)(v2 + 32) = v4;
  *(_QWORD *)(v2 + 48) = 0;
  if ( (*(_BYTE *)(a1 + 84) & 0x20) != 0 )
  {
    *(_QWORD *)(v2 + 24) = sub_87D520(a1);
    v5 = unk_4D03C48;
    if ( unk_4D03C48 )
      goto LABEL_5;
LABEL_8:
    unk_4D03C48 = v2;
    return v2;
  }
  v5 = unk_4D03C48;
  if ( !unk_4D03C48 )
    goto LABEL_8;
  do
  {
LABEL_5:
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 40);
  }
  while ( v5 );
  *(_QWORD *)(v6 + 40) = v2;
  return v2;
}
