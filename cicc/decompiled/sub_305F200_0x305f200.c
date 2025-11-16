// Function: sub_305F200
// Address: 0x305f200
//
__int64 __fastcall sub_305F200(__int64 a1, __int64 *a2)
{
  unsigned __int16 v2; // ax
  unsigned int v3; // r8d

  v2 = sub_2D5BAE0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), a2, 1);
  v3 = 0;
  if ( v2 )
    LOBYTE(v3) = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * v2 + 112) != 0;
  return v3;
}
