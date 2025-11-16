// Function: sub_2EC9A80
// Address: 0x2ec9a80
//
__int64 __fastcall sub_2EC9A80(__int64 *a1, char a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  bool v4; // dl
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // rax

  v2 = *a1;
  if ( *(_WORD *)(*a1 + 68) == 20 )
  {
    v3 = *(_QWORD *)(v2 + 32);
    if ( a2 )
    {
      if ( (unsigned int)(*(_DWORD *)(v3 + 48) - 1) <= 0x3FFFFFFE )
        return 1;
      v4 = *((_DWORD *)a1 + 55) == 0;
    }
    else
    {
      if ( (unsigned int)(*(_DWORD *)(v3 + 8) - 1) <= 0x3FFFFFFE )
        return 1;
      v4 = *((_DWORD *)a1 + 54) == 0;
      v3 += 40;
    }
    if ( (unsigned int)(*(_DWORD *)(v3 + 8) - 1) <= 0x3FFFFFFE )
    {
      if ( v4 )
        return 0xFFFFFFFFLL;
      return 1;
    }
  }
  result = 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v2 + 16) + 25LL) & 0x20) != 0 )
  {
    v6 = *(_QWORD *)(v2 + 32);
    v7 = v6 + 40LL * (unsigned int)sub_2E88FE0(*a1);
    v8 = *(_QWORD *)(v2 + 32);
    if ( v7 == v8 )
    {
LABEL_14:
      if ( a2 )
        return 0xFFFFFFFFLL;
      return 1;
    }
    while ( *(_BYTE *)v8 || (unsigned int)(*(_DWORD *)(v8 + 8) - 1) <= 0x3FFFFFFE )
    {
      v8 += 40;
      if ( v7 == v8 )
        goto LABEL_14;
    }
    return 0;
  }
  return result;
}
