// Function: sub_34B8590
// Address: 0x34b8590
//
__int64 __fastcall sub_34B8590(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d

  do
    v2 = sub_34B8410(a1, a2);
  while ( (_BYTE)v2
       && (unsigned __int8)(*(_BYTE *)(sub_B501B0(
                                         *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8),
                                         (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8) - 4),
                                         1)
                                     + 8)
                          - 15) <= 1u );
  return v2;
}
