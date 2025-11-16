// Function: sub_19306C0
// Address: 0x19306c0
//
__int64 __fastcall sub_19306C0(__int64 a1)
{
  char v1; // al
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD v13[4]; // [rsp-20h] [rbp-20h] BYREF

  v1 = *(_BYTE *)(a1 + 16);
  if ( (unsigned __int8)(v1 - 54) <= 1u )
    return 1;
  if ( v1 == 29 )
  {
    if ( !(unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 36) )
    {
      if ( *(char *)(a1 + 23) < 0 )
      {
        v8 = sub_1648A40(a1);
        v10 = v8 + v9;
        v11 = 0;
        if ( *(char *)(a1 + 23) < 0 )
          v11 = sub_1648A40(a1);
        if ( (unsigned int)((v10 - v11) >> 4) )
          return 1;
      }
      v12 = *(_QWORD *)(a1 - 72);
      if ( *(_BYTE *)(v12 + 16) )
        return 1;
      v13[0] = *(_QWORD *)(v12 + 112);
      if ( !(unsigned __int8)sub_1560260(v13, -1, 36) )
        return 1;
    }
    v1 = *(_BYTE *)(a1 + 16);
  }
  if ( v1 == 78 && !(unsigned __int8)sub_1560260((_QWORD *)(a1 + 56), -1, 36) )
  {
    if ( *(char *)(a1 + 23) >= 0 )
      goto LABEL_25;
    v3 = sub_1648A40(a1);
    v5 = v3 + v4;
    v6 = 0;
    if ( *(char *)(a1 + 23) < 0 )
      v6 = sub_1648A40(a1);
    if ( !(unsigned int)((v5 - v6) >> 4) )
    {
LABEL_25:
      v7 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v7 + 16) )
      {
        v13[0] = *(_QWORD *)(v7 + 112);
        return (unsigned int)sub_1560260(v13, -1, 36) ^ 1;
      }
    }
    return 1;
  }
  return 0;
}
