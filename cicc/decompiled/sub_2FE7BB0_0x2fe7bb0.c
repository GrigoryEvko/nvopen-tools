// Function: sub_2FE7BB0
// Address: 0x2fe7bb0
//
__int64 __fastcall sub_2FE7BB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 *v3; // rdx
  __int64 v4; // rax

  v3 = (unsigned __int16 *)(*(_QWORD *)(a2 + 320)
                          + 2LL
                          * *(unsigned int *)(*(_QWORD *)(a2 + 312)
                                            + 16LL
                                            * (*(unsigned __int16 *)(*(_QWORD *)a3 + 24LL)
                                             + *(_DWORD *)(a2 + 328)
                                             * (unsigned int)((__int64)(*(_QWORD *)(a2 + 288) - *(_QWORD *)(a2 + 280)) >> 3))
                                            + 12));
  v4 = *v3;
  if ( (_WORD)v4 == 1 )
    return 0;
  while ( !(_WORD)v4 || !*(_QWORD *)(a1 + 8 * v4 + 112) )
  {
    v4 = v3[1];
    ++v3;
    if ( (_WORD)v4 == 1 )
      return 0;
  }
  return 1;
}
