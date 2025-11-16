// Function: sub_2F7E810
// Address: 0x2f7e810
//
__int64 __fastcall sub_2F7E810(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // rax

  if ( !a1 )
    return 0;
  v2 = a1;
  do
  {
    v3 = (__int64 *)sub_2E09D00((__int64 *)v2, a2);
    if ( v3 != (__int64 *)(*(_QWORD *)v2 + 24LL * *(unsigned int *)(v2 + 8))
      && (*(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v3 >> 1) & 3) <= ((unsigned int)(a2 >> 1) & 3
                                                                                           | *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    {
      return 1;
    }
    v2 = *(_QWORD *)(v2 + 104);
  }
  while ( v2 );
  return 0;
}
