// Function: sub_1EB6550
// Address: 0x1eb6550
//
__int64 __fastcall sub_1EB6550(__int64 *a1, int a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rax

  v3 = a2 & 0x7FFFFFFF;
  result = *(unsigned int *)(a1[46] + 4 * v3);
  if ( (_DWORD)result == -1 )
  {
    v5 = *(_QWORD *)(a1[31] + 280)
       + 24LL
       * (*(unsigned __int16 *)(*(_QWORD *)a3 + 24LL)
        + *(_DWORD *)(a1[31] + 288)
        * (unsigned int)((__int64)(*(_QWORD *)(a1[31] + 264) - *(_QWORD *)(a1[31] + 256)) >> 3));
    result = sub_1E091A0(a1[29], *(_DWORD *)(v5 + 4) >> 3, *(_DWORD *)(v5 + 8) >> 3);
    *(_DWORD *)(a1[46] + 4 * v3) = result;
  }
  return result;
}
