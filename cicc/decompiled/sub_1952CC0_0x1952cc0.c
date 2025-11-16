// Function: sub_1952CC0
// Address: 0x1952cc0
//
__int64 __fastcall sub_1952CC0(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v1) + 16) - 25) > 9u )
    {
      v1 = *(_QWORD *)(v1 + 8);
      if ( !v1 )
        return 0;
    }
  }
  return v1;
}
