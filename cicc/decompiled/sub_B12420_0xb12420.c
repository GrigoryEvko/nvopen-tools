// Function: sub_B12420
// Address: 0xb12420
//
__int64 __fastcall sub_B12420(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  char v3; // al
  __int64 v4; // r12
  unsigned int v6; // r13d

  v3 = *(_BYTE *)(a1 + 32);
  if ( v3 != *(_BYTE *)(a2 + 32) )
    return 0;
  if ( !v3 )
  {
    v6 = 0;
    if ( *(_BYTE *)(a1 + 64) == *(_BYTE *)(a2 + 64)
      && !memcmp((const void *)(a1 + 40), (const void *)(a2 + 40), 0x18u)
      && *(_QWORD *)(a1 + 72) == *(_QWORD *)(a2 + 72)
      && *(_QWORD *)(a1 + 80) == *(_QWORD *)(a2 + 80) )
    {
      LOBYTE(v6) = *(_QWORD *)(a1 + 88) == *(_QWORD *)(a2 + 88);
    }
    return v6;
  }
  if ( v3 != 1 )
    BUG();
  v4 = sub_B11FB0(a1 + 40);
  LOBYTE(v2) = v4 == sub_B11FB0(a2 + 40);
  return v2;
}
