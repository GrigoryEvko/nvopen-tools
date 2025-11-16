// Function: sub_AA54C0
// Address: 0xaa54c0
//
__int64 __fastcall sub_AA54C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8

  v1 = *(_QWORD *)(a1 + 16);
  while ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 24);
    v1 = *(_QWORD *)(v1 + 8);
    if ( (unsigned __int8)(*(_BYTE *)v2 - 30) <= 0xAu )
    {
      v3 = *(_QWORD *)(v2 + 40);
      if ( !v1 )
        return v3;
      while ( (unsigned __int8)(**(_BYTE **)(v1 + 24) - 30) > 0xAu )
      {
        v1 = *(_QWORD *)(v1 + 8);
        if ( !v1 )
          return *(_QWORD *)(v2 + 40);
      }
      return 0;
    }
  }
  return 0;
}
