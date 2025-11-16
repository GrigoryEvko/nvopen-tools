// Function: sub_140AFC0
// Address: 0x140afc0
//
__int64 __fastcall sub_140AFC0(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v3; // r13
  unsigned int v5; // eax
  unsigned int v6; // r12d
  unsigned __int8 v8; // al
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1;
  v5 = sub_140AF60(a1, a2, a3);
  if ( (_BYTE)v5 )
    return 1;
  v6 = v5;
  if ( a3 )
    v3 = sub_1649C60(a1);
  v8 = *(_BYTE *)(v3 + 16);
  if ( v8 > 0x17u )
  {
    if ( v8 == 78 )
    {
      v9 = v3 | 4;
    }
    else
    {
      if ( v8 != 29 )
        return v6;
      v9 = v3 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      return v6;
    v11 = v10 + 56;
    if ( (v9 & 4) != 0 )
    {
      if ( !(unsigned __int8)sub_1560260(v11, 0, 20) )
      {
        v12 = *(_QWORD *)(v10 - 24);
        if ( *(_BYTE *)(v12 + 16) )
          return v6;
LABEL_18:
        v13[0] = *(_QWORD *)(v12 + 112);
        if ( !(unsigned __int8)sub_1560260(v13, 0, 20) )
          return v6;
      }
    }
    else if ( !(unsigned __int8)sub_1560260(v11, 0, 20) )
    {
      v12 = *(_QWORD *)(v10 - 72);
      if ( *(_BYTE *)(v12 + 16) )
        return v6;
      goto LABEL_18;
    }
    return 1;
  }
  return v6;
}
