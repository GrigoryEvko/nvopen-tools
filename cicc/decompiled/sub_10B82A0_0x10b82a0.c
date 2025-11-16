// Function: sub_10B82A0
// Address: 0x10b82a0
//
bool __fastcall sub_10B82A0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  bool result; // al
  __int64 v5; // rdx
  unsigned int v6; // edx

  result = 0;
  if ( *(_BYTE *)a1 == 83 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    if ( v5 )
    {
      if ( !*(_QWORD *)(v5 + 8) )
      {
        *a2 = sub_98A030(
                *(_WORD *)(a1 + 2) & 0x3F,
                *(_QWORD *)(*(_QWORD *)(a1 + 40) + 72LL),
                *(_QWORD *)(a1 - 64),
                *(_QWORD *)(a1 - 32),
                1);
        *a3 = v6;
        return *a2 != 0;
      }
    }
  }
  return result;
}
