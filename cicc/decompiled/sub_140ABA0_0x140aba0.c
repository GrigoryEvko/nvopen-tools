// Function: sub_140ABA0
// Address: 0x140aba0
//
__int64 __fastcall sub_140ABA0(__int64 a1, char a2, _BYTE *a3)
{
  unsigned __int8 v4; // al
  __int64 result; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // r13
  char v9; // bl
  unsigned __int64 v10; // rdi
  int v11; // ebx
  __int64 v12; // rax
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 == 78 )
  {
    v6 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v6 + 16) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
      return 0;
    if ( !a2 )
    {
LABEL_11:
      v7 = a1 | 4;
      goto LABEL_12;
    }
  }
  else if ( !a2 )
  {
    if ( v4 <= 0x17u )
      return 0;
    goto LABEL_4;
  }
  a1 = sub_1649C60(a1);
  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 <= 0x17u )
    return 0;
  if ( v4 == 78 )
    goto LABEL_11;
LABEL_4:
  if ( v4 != 29 )
    return 0;
  v7 = a1 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_12:
  v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v9 = v7 >> 2;
  v10 = v8 + 56;
  v11 = v9 & 1;
  if ( v11 )
  {
    if ( !(unsigned __int8)sub_1560260(v10, 0xFFFFFFFFLL, 21)
      && ((v12 = *(_QWORD *)(v8 - 24), *(_BYTE *)(v12 + 16))
       || (v17[0] = *(_QWORD *)(v12 + 112), !(unsigned __int8)sub_1560260(v17, 0xFFFFFFFFLL, 21)))
      || (unsigned __int8)sub_1560260(v8 + 56, 0xFFFFFFFFLL, 5) )
    {
      LOBYTE(v11) = 0;
    }
    else
    {
      v14 = *(_QWORD *)(v8 - 24);
      if ( !*(_BYTE *)(v14 + 16) )
      {
        v17[0] = *(_QWORD *)(v14 + 112);
        LOBYTE(v11) = sub_1560260(v17, 0xFFFFFFFFLL, 5) ^ 1;
      }
    }
    *a3 = v11;
    v13 = v8 - 24;
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v10, 0xFFFFFFFFLL, 21)
      || (v15 = *(_QWORD *)(v8 - 72), !*(_BYTE *)(v15 + 16))
      && (v17[0] = *(_QWORD *)(v15 + 112), (unsigned __int8)sub_1560260(v17, 0xFFFFFFFFLL, 21)) )
    {
      if ( !(unsigned __int8)sub_1560260(v8 + 56, 0xFFFFFFFFLL, 5) )
      {
        v16 = *(_QWORD *)(v8 - 72);
        LOBYTE(v11) = 1;
        if ( !*(_BYTE *)(v16 + 16) )
        {
          v17[0] = *(_QWORD *)(v16 + 112);
          LOBYTE(v11) = sub_1560260(v17, 0xFFFFFFFFLL, 5) ^ 1;
        }
      }
    }
    *a3 = v11;
    v13 = v8 - 72;
  }
  result = *(_QWORD *)v13;
  if ( *(_BYTE *)(*(_QWORD *)v13 + 16LL) )
    return 0;
  return result;
}
