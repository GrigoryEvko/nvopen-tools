// Function: sub_85ED80
// Address: 0x85ed80
//
__int64 __fastcall sub_85ED80(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rax
  unsigned int v6; // r8d

  v2 = *(_BYTE *)(a2 + 4);
  v3 = 1;
  if ( !v2 )
    return v3;
  if ( (unsigned __int8)(v2 - 3) > 1u )
    return 0;
  v5 = sub_8807C0(a1);
  v6 = 0;
  if ( v5 )
  {
    while ( *(_QWORD *)(*(_QWORD *)(a2 + 184) + 32LL) != v5 && v5 )
    {
      v5 = *(_QWORD *)(v5 + 40);
      if ( v5 )
      {
        if ( *(_BYTE *)(v5 + 28) == 3 )
          v5 = *(_QWORD *)(v5 + 32);
        else
          v5 = 0;
      }
    }
    return v5 != 0;
  }
  return v6;
}
