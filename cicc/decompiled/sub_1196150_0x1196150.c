// Function: sub_1196150
// Address: 0x1196150
//
__int64 __fastcall sub_1196150(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *(_QWORD *)(a2 + 16);
  v3 = 0;
  if ( v2 )
  {
    if ( !*(_QWORD *)(v2 + 8) && *(_BYTE *)a2 == 54 )
    {
      v5 = *(_QWORD *)(a2 - 64);
      if ( v5 )
      {
        **(_QWORD **)a1 = v5;
        LOBYTE(v3) = *(_QWORD *)(a2 - 32) == *(_QWORD *)(a1 + 8);
      }
    }
  }
  return v3;
}
