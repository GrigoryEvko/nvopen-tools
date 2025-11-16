// Function: sub_19E4680
// Address: 0x19e4680
//
__int64 __fastcall sub_19E4680(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx

  v2 = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)(a1 + 16) == v2 )
  {
    v6 = &v2[*(unsigned int *)(a1 + 28)];
    if ( v2 == v6 )
    {
LABEL_13:
      v2 = v6;
    }
    else
    {
      while ( a2 != *v2 )
      {
        if ( v6 == ++v2 )
          goto LABEL_13;
      }
    }
  }
  else
  {
    v2 = sub_16CC9F0(a1, a2);
    v3 = *(_QWORD *)(a1 + 16);
    if ( a2 == *v2 )
    {
      if ( *(_QWORD *)(a1 + 8) == v3 )
        v5 = *(unsigned int *)(a1 + 28);
      else
        v5 = *(unsigned int *)(a1 + 24);
      v6 = (_QWORD *)(v3 + 8 * v5);
    }
    else
    {
      result = 0;
      if ( v3 != *(_QWORD *)(a1 + 8) )
        return result;
      v2 = (_QWORD *)(v3 + 8LL * *(unsigned int *)(a1 + 28));
      v6 = v2;
    }
  }
  if ( v6 == v2 )
    return 0;
  *v2 = -2;
  ++*(_DWORD *)(a1 + 32);
  return 1;
}
