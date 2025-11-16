// Function: sub_1268C40
// Address: 0x1268c40
//
__int64 __fastcall sub_1268C40(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  char v3; // al
  unsigned __int8 v5; // al
  int v6; // [rsp+4h] [rbp-1Ch] BYREF
  int v7; // [rsp+8h] [rbp-18h] BYREF
  _BOOL4 v8; // [rsp+Ch] [rbp-14h] BYREF

  v6 = 0;
  v7 = 0;
  v8 = 0;
  if ( (*(_BYTE *)(a1 + 198) & 0x10) != 0 )
  {
    sub_8267B0(a1, &v6, &v7, &v8);
    v2 = 3;
    if ( v8 )
      goto LABEL_3;
    v2 = 5;
    if ( v7 )
      goto LABEL_3;
    if ( v6 )
    {
      v2 = 4;
      goto LABEL_3;
    }
  }
  v5 = *(_BYTE *)(a1 + 172);
  v2 = 0;
  if ( v5 > 1u )
  {
    v2 = 7;
    if ( v5 != 2 )
      sub_127B550("unsupported storage class!");
  }
LABEL_3:
  v3 = *(_BYTE *)(a1 + 198);
  if ( (v3 & 0x10) != 0
    && (v3 & 0x20) == 0
    && (!a2 || (_DWORD)qword_4D045BC)
    && !HIDWORD(qword_4D045BC)
    && *(_DWORD *)(a1 + 160)
    && !(unk_4D04630 | unk_4D04614) )
  {
    return 7;
  }
  return v2;
}
