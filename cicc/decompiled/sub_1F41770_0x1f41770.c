// Function: sub_1F41770
// Address: 0x1f41770
//
__int64 __fastcall sub_1F41770(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // rdx
  __int64 v4; // rax

  v3 = *(unsigned __int8 **)(*(_QWORD *)(a2 + 280)
                           + 24LL
                           * (*(unsigned __int16 *)(*(_QWORD *)a3 + 24LL)
                            + *(_DWORD *)(a2 + 288)
                            * (unsigned int)((__int64)(*(_QWORD *)(a2 + 264) - *(_QWORD *)(a2 + 256)) >> 3))
                           + 16);
  v4 = *v3;
  if ( (_BYTE)v4 == 1 )
    return 0;
  while ( !(_BYTE)v4 || !*(_QWORD *)(a1 + 8 * v4 + 120) )
  {
    v4 = *++v3;
    if ( (_BYTE)v4 == 1 )
      return 0;
  }
  return 1;
}
