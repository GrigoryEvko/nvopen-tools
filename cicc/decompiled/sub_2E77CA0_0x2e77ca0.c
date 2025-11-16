// Function: sub_2E77CA0
// Address: 0x2e77ca0
//
__int64 __fastcall sub_2E77CA0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned __int8 v3; // r13
  unsigned int v4; // r14d

  v3 = a3;
  if ( a3 > (unsigned int)*(_WORD *)a1 )
    v3 = *(_BYTE *)a1;
  sub_2E77BD0(a1, a2, v3, 1u, 0, 0);
  v4 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) + ~*(_DWORD *)(a1 + 32);
  sub_2E76F70(a1, v3);
  return v4;
}
