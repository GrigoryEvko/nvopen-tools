// Function: sub_B1A070
// Address: 0xb1a070
//
char __fastcall sub_B1A070(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // edx
  char result; // al

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 <= 0x1Cu )
    return 1;
  if ( *(_BYTE *)v2 != 84 )
    return sub_B192B0(a1, *(_QWORD *)(v2 + 40));
  v3 = *(_QWORD *)(*(_QWORD *)(v2 - 8)
                 + 32LL * *(unsigned int *)(v2 + 72)
                 + 8LL * (unsigned int)((a2 - *(_QWORD *)(v2 - 8)) >> 5));
  if ( v3 )
  {
    v4 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
    v5 = *(_DWORD *)(v3 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  result = 0;
  if ( v5 < *(_DWORD *)(a1 + 32) )
    return *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v4) != 0;
  return result;
}
