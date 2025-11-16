// Function: sub_1EECF40
// Address: 0x1eecf40
//
__int64 __fastcall sub_1EECF40(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // rax

  v2 = a1;
  if ( !a1 )
    return 0;
  while ( 1 )
  {
    v3 = (__int64 *)sub_1DB3C70((__int64 *)v2, a2);
    if ( v3 != (__int64 *)(*(_QWORD *)v2 + 24LL * *(unsigned int *)(v2 + 8))
      && (*(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v3 >> 1) & 3) <= ((unsigned int)(a2 >> 1) & 3
                                                                                           | *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    {
      break;
    }
    v2 = *(_QWORD *)(v2 + 104);
    if ( !v2 )
      return 0;
  }
  return 1;
}
