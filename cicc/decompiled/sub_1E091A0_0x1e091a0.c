// Function: sub_1E091A0
// Address: 0x1e091a0
//
__int64 __fastcall sub_1E091A0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // r13d
  unsigned int v4; // r14d

  v3 = *(_DWORD *)a1;
  if ( *(_BYTE *)(a1 + 4) || v3 >= a3 )
    v3 = a3;
  sub_1E090F0(a1, a2, v3, 1, 0, 0);
  v4 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) + ~*(_DWORD *)(a1 + 32);
  sub_1E08740(a1, v3);
  return v4;
}
