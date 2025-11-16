// Function: sub_1B974E0
// Address: 0x1b974e0
//
__int64 __fastcall sub_1B974E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r12

  if ( *(char *)(a1 + 23) >= 0 )
    return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v1 = sub_1648A40(a1);
  v3 = v1 + v2;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( !(unsigned int)(v3 >> 4) )
      return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    goto LABEL_9;
  }
  if ( !(unsigned int)((v3 - sub_1648A40(a1)) >> 4) )
    return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( *(char *)(a1 + 23) >= 0 )
LABEL_9:
    BUG();
  sub_1648A40(a1);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  sub_1648A40(a1);
  return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
}
