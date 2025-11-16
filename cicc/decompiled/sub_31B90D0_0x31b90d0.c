// Function: sub_31B90D0
// Address: 0x31b90d0
//
__int64 __fastcall sub_31B90D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r13
  char v4; // r8
  __int64 result; // rax
  char v6; // r8
  unsigned __int8 *v7; // rcx
  int v8; // edx
  __int64 v9; // rax
  char v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rsi

  v2 = *(_QWORD *)(a1 + 16);
  if ( (unsigned __int8)sub_B46490(v2) )
  {
    v3 = *(_QWORD *)(a2 + 16);
    v4 = sub_B46420(v3);
    result = 0;
    if ( v4 )
      return result;
    v6 = sub_B46490(v3);
    result = 1;
    if ( v6 )
      return result;
LABEL_4:
    if ( !sub_318B700(a1) )
      goto LABEL_5;
    return 3;
  }
  if ( !(unsigned __int8)sub_B46420(v2) )
    goto LABEL_4;
  v10 = sub_B46490(*(_QWORD *)(a2 + 16));
  result = 2;
  if ( v10 )
    return result;
  if ( sub_318B700(a1) )
    return 3;
LABEL_5:
  if ( sub_318B700(a2) )
    return 3;
  v7 = *(unsigned __int8 **)(a2 + 16);
  v8 = *v7;
  if ( (unsigned int)(v8 - 30) <= 0xA )
    return 3;
  v9 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)v9 != 85
    || (v12 = *(_QWORD *)(v9 - 32)) == 0
    || *(_BYTE *)v12
    || *(_QWORD *)(v12 + 24) != *(_QWORD *)(v9 + 80)
    || (*(_BYTE *)(v12 + 33) & 0x20) == 0
    || (result = 4, (unsigned int)(*(_DWORD *)(v12 + 36) - 342) > 1) )
  {
    result = 5;
    if ( (_BYTE)v8 == 85 )
    {
      v11 = *((_QWORD *)v7 - 4);
      if ( v11 )
      {
        if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *((_QWORD *)v7 + 10) && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
          return (unsigned int)((unsigned int)(*(_DWORD *)(v11 + 36) - 342) > 1) + 4;
      }
    }
  }
  return result;
}
