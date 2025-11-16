// Function: sub_5F2840
// Address: 0x5f2840
//
__int64 __fastcall sub_5F2840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  char v5; // al
  __int64 v6; // rax
  char v7; // dl
  __int64 v9; // rax

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 140);
  if ( v5 == 12 )
  {
    v6 = a2;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v7 = *(_BYTE *)(v6 + 140);
    }
    while ( v7 == 12 );
    if ( !v7 )
      return v4;
    goto LABEL_8;
  }
  if ( !v5 )
    return v4;
  if ( (v5 & 0xFB) == 8 )
  {
LABEL_8:
    if ( (sub_8D4C10(a2, dword_4F077C4 != 2) & 1) == 0 )
      goto LABEL_9;
    sub_685360(1591, a3);
    if ( *(char *)(a1 + 124) < 0 )
    {
      v4 = sub_72C930();
      v9 = sub_72C930();
      *(_QWORD *)(a1 + 272) = v9;
      *(_QWORD *)(a1 + 280) = v9;
      *(_QWORD *)(a1 + 288) = v9;
    }
    return v4;
  }
LABEL_9:
  sub_6851C0(1592, a3);
  return a2;
}
