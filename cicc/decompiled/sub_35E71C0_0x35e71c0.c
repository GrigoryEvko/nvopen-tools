// Function: sub_35E71C0
// Address: 0x35e71c0
//
__int64 __fastcall sub_35E71C0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rcx
  __int64 v6; // rdx

  result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL);
  if ( a2 )
  {
    v4 = a2;
    while ( result )
    {
      if ( (*(_BYTE *)result & 4) != 0 )
      {
        result = *(_QWORD *)(result + 8);
        if ( !--v4 )
          goto LABEL_7;
      }
      else
      {
        while ( (*(_BYTE *)(result + 44) & 8) != 0 )
          result = *(_QWORD *)(result + 8);
        result = *(_QWORD *)(result + 8);
        if ( !--v4 )
          goto LABEL_7;
      }
    }
LABEL_21:
    BUG();
  }
LABEL_7:
  if ( a3 )
  {
    v5 = a3;
    v6 = result;
    while ( v6 )
    {
      if ( (*(_BYTE *)v6 & 4) != 0 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !--v5 )
          return result;
      }
      else
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        if ( !--v5 )
          return result;
      }
    }
    goto LABEL_21;
  }
  return result;
}
