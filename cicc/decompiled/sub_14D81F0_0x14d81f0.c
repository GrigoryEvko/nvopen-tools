// Function: sub_14D81F0
// Address: 0x14d81f0
//
__int64 __fastcall sub_14D81F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // ebx
  unsigned int i; // r13d

  if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    return 0;
  v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v3 != 2 )
  {
    v4 = v3 - 1;
    for ( i = 2; ; ++i )
    {
      a1 = sub_15A0F90(a1, *(_QWORD *)(a2 + 24 * (i - v3)));
      if ( !a1 )
        break;
      if ( v4 == i )
        return a1;
      v3 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    return 0;
  }
  return a1;
}
