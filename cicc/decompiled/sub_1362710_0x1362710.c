// Function: sub_1362710
// Address: 0x1362710
//
__int64 *__fastcall sub_1362710(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v5; // eax
  int v6; // eax
  unsigned int v7; // esi
  __int64 *v9; // [rsp+8h] [rbp-18h] BYREF

  v5 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v6 = (v5 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
  {
    v7 = *(_DWORD *)(a1 + 24);
    if ( 4 * v6 < 3 * v7 )
      goto LABEL_3;
LABEL_8:
    v7 *= 2;
    goto LABEL_9;
  }
  v7 = 8;
  if ( (unsigned int)(4 * v6) >= 0x18 )
    goto LABEL_8;
LABEL_3:
  if ( v7 - (v6 + *(_DWORD *)(a1 + 12)) <= v7 >> 3 )
  {
LABEL_9:
    sub_1362490(a1, v7);
    sub_1361B70(a1, a2, &v9);
    a3 = v9;
    v6 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v6);
  if ( *a3 != -8 || a3[1] || a3[2] || a3[3] || a3[4] || a3[5] != -8 || a3[6] || a3[7] || a3[8] || a3[9] )
    --*(_DWORD *)(a1 + 12);
  return a3;
}
