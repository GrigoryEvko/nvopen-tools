// Function: sub_8D1AD0
// Address: 0x8d1ad0
//
__int64 __fastcall sub_8D1AD0(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // r9

  if ( *(_BYTE *)(a1 + 140) != 8 || (*(_BYTE *)(a1 + 169) & 0x10) == 0 )
    return 0;
  v5 = sub_72D900(a1)[2];
  if ( !v5 || !(unsigned int)sub_731770(v5, 0, v3, v4, v6, v7) )
    return 0;
  *a2 = 1;
  return 1;
}
