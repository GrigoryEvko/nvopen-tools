// Function: sub_157F0B0
// Address: 0x157f0b0
//
__int64 __fastcall sub_157F0B0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r12

  v1 = *(_QWORD *)(a1 + 8);
  while ( v1 )
  {
    v2 = sub_1648700(v1);
    v1 = *(_QWORD *)(v1 + 8);
    if ( (unsigned __int8)(*(_BYTE *)(v2 + 16) - 25) <= 9u )
    {
      v3 = *(_QWORD *)(v2 + 40);
      if ( !v1 )
        return v3;
      while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v1) + 16) - 25) > 9u )
      {
        v1 = *(_QWORD *)(v1 + 8);
        if ( !v1 )
          return v3;
      }
      return 0;
    }
  }
  return 0;
}
