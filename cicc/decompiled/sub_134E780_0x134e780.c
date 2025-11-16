// Function: sub_134E780
// Address: 0x134e780
//
__int64 __fastcall sub_134E780(__int64 a1)
{
  __int64 v1; // rbp
  unsigned __int8 v2; // al
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rbx
  bool v6; // zf
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD v9[4]; // [rsp-20h] [rbp-20h] BYREF

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 <= 0x17u )
    return 0;
  if ( v2 == 78 )
  {
    v4 = a1 | 4;
  }
  else
  {
    if ( v2 != 29 )
      return 0;
    v4 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v9[3] = v1;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v6 = (v4 & 4) == 0;
  v7 = v5 + 56;
  if ( !v6 )
  {
    if ( (unsigned __int8)sub_1560260(v7, 0, 20) )
      return 1;
    v8 = *(_QWORD *)(v5 - 24);
    if ( !*(_BYTE *)(v8 + 16) )
      goto LABEL_10;
    return 0;
  }
  if ( !(unsigned __int8)sub_1560260(v7, 0, 20) )
  {
    v8 = *(_QWORD *)(v5 - 72);
    if ( !*(_BYTE *)(v8 + 16) )
    {
LABEL_10:
      v9[0] = *(_QWORD *)(v8 + 112);
      return (unsigned int)sub_1560260(v9, 0, 20);
    }
    return 0;
  }
  return 1;
}
