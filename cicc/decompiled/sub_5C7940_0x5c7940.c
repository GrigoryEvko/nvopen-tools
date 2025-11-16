// Function: sub_5C7940
// Address: 0x5c7940
//
__int64 __fastcall sub_5C7940(__int64 a1)
{
  int v1; // r12d
  __int16 v2; // bx
  __int64 v3; // rax
  char i; // dl
  __int64 result; // rax
  __int64 v6; // rdx
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = unk_4F063F8;
  v2 = unk_4F063FC;
  sub_65CD60(v7);
  v3 = v7[0];
  for ( i = *(_BYTE *)(v7[0] + 140LL); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  if ( i )
  {
    result = sub_7276D0();
    *(_BYTE *)(result + 10) = 4;
    v6 = unk_4F061D8;
    *(_DWORD *)(result + 24) = v1;
    *(_WORD *)(result + 28) = v2;
    *(_QWORD *)(result + 32) = v6;
    *(_QWORD *)(result + 40) = v7[0];
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
    return 0;
  }
  return result;
}
