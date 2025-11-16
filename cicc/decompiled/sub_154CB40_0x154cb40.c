// Function: sub_154CB40
// Address: 0x154cb40
//
__int64 __fastcall sub_154CB40(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rbx
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rdx

  if ( *(char *)(a1 + 23) >= 0 )
    return 0;
  v1 = sub_1648A40(a1);
  v3 = v1 + v2;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v3 >> 4) )
LABEL_10:
      BUG();
    return 0;
  }
  if ( !(unsigned int)((v3 - sub_1648A40(a1)) >> 4) )
    return 0;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_10;
  v4 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v5 = sub_1648A40(a1);
  return (unsigned int)(*(_DWORD *)(v5 + v6 - 4) - v4);
}
