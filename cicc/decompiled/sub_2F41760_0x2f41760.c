// Function: sub_2F41760
// Address: 0x2f41760
//
__int64 __fastcall sub_2F41760(__int64 *a1, int a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  unsigned __int8 v4; // r8
  __int64 v5; // rsi
  __int64 v6; // rax

  v2 = a2 & 0x7FFFFFFF;
  result = *(unsigned int *)(a1[49] + 4 * v2);
  if ( (_DWORD)result == -1 )
  {
    v4 = -1;
    v6 = *(_QWORD *)(a1[2] + 312)
       + 16LL
       * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[1] + 56) + 16 * v2) & 0xFFFFFFFFFFFFFFF8LL) + 24LL)
        + *(_DWORD *)(a1[2] + 328) * (unsigned int)((__int64)(*(_QWORD *)(a1[2] + 288) - *(_QWORD *)(a1[2] + 280)) >> 3));
    v5 = *(_DWORD *)(v6 + 4) >> 3;
    LODWORD(v6) = *(_DWORD *)(v6 + 8) >> 3;
    if ( (_DWORD)v6 )
    {
      _BitScanReverse64((unsigned __int64 *)&v6, (unsigned int)v6);
      v4 = 63 - (v6 ^ 0x3F);
    }
    result = sub_2E77CA0(*a1, v5, v4);
    *(_DWORD *)(a1[49] + 4 * v2) = result;
  }
  return result;
}
