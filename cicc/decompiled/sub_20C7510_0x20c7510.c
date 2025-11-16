// Function: sub_20C7510
// Address: 0x20c7510
//
__int64 __fastcall sub_20C7510(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d

  do
    v2 = sub_20C73B0(a1, a2);
  while ( (_BYTE)v2
       && (unsigned int)*(unsigned __int8 *)(sub_1643D80(
                                               *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8),
                                               *(_DWORD *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8) - 4))
                                           + 8)
        - 13 <= 1 );
  return v2;
}
