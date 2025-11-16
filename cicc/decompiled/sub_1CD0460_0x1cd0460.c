// Function: sub_1CD0460
// Address: 0x1cd0460
//
__int64 __fastcall sub_1CD0460(__int64 a1, __int64 a2)
{
  char v2; // dl
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  unsigned int v6; // edx
  __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // edx

  v2 = *(_BYTE *)(a2 + 16);
  if ( !*(_DWORD *)(a1 + 4) )
    goto LABEL_6;
  result = 0;
  if ( v2 == 71 )
    return result;
  if ( v2 == 56 )
  {
    v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( a2 == a2 + 24 * (1 - v4) )
      return 1;
    else
      return (unsigned int)((unsigned __int64)(-24 - 24 * (1 - v4)) >> 3) + 4;
  }
  else
  {
LABEL_6:
    if ( v2 == 54 )
    {
      v5 = **(_QWORD **)(a2 - 24);
      if ( *(_BYTE *)(v5 + 8) == 16 )
        v5 = **(_QWORD **)(v5 + 16);
      v6 = *(_DWORD *)(v5 + 8) >> 8;
      if ( v6 == 5 )
        return 8;
      result = 2;
      if ( v6 <= 1 )
        return 8;
    }
    else if ( v2 == 55 )
    {
      v7 = **(_QWORD **)(a2 - 24);
      if ( *(_BYTE *)(v7 + 8) == 16 )
        v7 = **(_QWORD **)(v7 + 16);
      v8 = *(_DWORD *)(v7 + 8);
      v9 = v8 >> 8;
      if ( v8 <= 0x1FF )
        return 10;
      result = 2;
      if ( v9 == 5 )
        return 10;
    }
    else
    {
      return (unsigned __int8)(v2 - 41) < 6u ? 5 : 1;
    }
  }
  return result;
}
