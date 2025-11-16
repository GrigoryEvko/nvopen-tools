// Function: sub_600680
// Address: 0x600680
//
__int64 __fastcall sub_600680(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // [rsp+8h] [rbp-18h] BYREF
  _BYTE v10[20]; // [rsp+Ch] [rbp-14h] BYREF

  v2 = a1;
  if ( (unsigned int)sub_8D3410(a1) )
    v2 = sub_8D40F0(a1);
  while ( 1 )
  {
    v3 = *(_BYTE *)(v2 + 140);
    if ( v3 != 12 )
      break;
    v2 = *(_QWORD *)(v2 + 160);
  }
  if ( (unsigned __int8)(v3 - 9) > 2u )
  {
    result = 0;
LABEL_7:
    if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
      return *(_BYTE *)(a2 + 140) != 11;
    return result;
  }
  sub_600530(v2);
  v5 = sub_87CAB0(v2, (unsigned int)dword_4F07508, a2, 1, 1, 1, 0, (__int64)&v9, (__int64)v10);
  if ( v5 )
    return (*(_BYTE *)(v5 + 193) & 3) != 0;
  v6 = *(_QWORD *)(*(_QWORD *)v2 + 96LL);
  result = 0;
  if ( (*(_BYTE *)(v6 + 176) & 1) == 0 && (*(_QWORD *)(v6 + 16) || !*(_QWORD *)(v6 + 8)) )
  {
    v7 = *(_QWORD *)(v6 + 24);
    if ( !v7
      || (*(_BYTE *)(v6 + 177) & 2) != 0
      || (v8 = *(_QWORD *)(v7 + 88), result = 0, (*(_BYTE *)(v8 + 193) & 2) != 0) )
    {
      result = v9;
      if ( v9 )
        return 0;
      if ( (*(_BYTE *)(v2 + 179) & 1) != 0 )
        return 1;
      goto LABEL_7;
    }
  }
  return result;
}
