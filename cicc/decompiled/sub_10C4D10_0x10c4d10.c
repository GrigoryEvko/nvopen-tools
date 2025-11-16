// Function: sub_10C4D10
// Address: 0x10c4d10
//
__int64 __fastcall sub_10C4D10(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 57 )
    {
      v3 = 1;
      if ( *(_QWORD *)(a2 - 64) != *a1 )
        LOBYTE(v3) = *(_QWORD *)(a2 - 32) == *a1;
    }
  }
  return v3;
}
