// Function: sub_250CB50
// Address: 0x250cb50
//
__int64 __fastcall sub_250CB50(__int64 *a1, char a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rax

  if ( a2 )
  {
    v4 = sub_250C680(a1);
    if ( v4 )
      return *(unsigned int *)(v4 + 32);
  }
  v2 = *a1;
  v3 = *a1 & 3;
  if ( v3 != 3 )
  {
    if ( v3 == 2 )
      return 0xFFFFFFFFLL;
    v4 = v2 & 0xFFFFFFFFFFFFFFFCLL;
    if ( !v4 || *(_BYTE *)v4 != 22 )
      return 0xFFFFFFFFLL;
    return *(unsigned int *)(v4 + 32);
  }
  return (unsigned int)((__int64)((v2 & 0xFFFFFFFFFFFFFFFCLL)
                                - (*(_QWORD *)((v2 & 0xFFFFFFFFFFFFFFFCLL) + 24)
                                 - 32LL * (*(_DWORD *)(*(_QWORD *)((v2 & 0xFFFFFFFFFFFFFFFCLL) + 24) + 4LL) & 0x7FFFFFF))) >> 5);
}
