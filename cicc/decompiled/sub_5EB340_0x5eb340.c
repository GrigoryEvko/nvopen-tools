// Function: sub_5EB340
// Address: 0x5eb340
//
__int64 __fastcall sub_5EB340(__int64 a1)
{
  __int64 result; // rax
  char v2; // cl
  char v3; // dl
  __int64 v4; // rdx
  __int64 i; // rdx

  result = *(_QWORD *)(a1 + 8);
  if ( !result )
    return *(_QWORD *)(a1 + 16);
  v2 = *(_BYTE *)(result + 80);
  v3 = v2;
  if ( v2 != 17 )
    goto LABEL_11;
  for ( result = *(_QWORD *)(result + 88); result; result = *(_QWORD *)(result + 8) )
  {
    v3 = *(_BYTE *)(result + 80);
LABEL_11:
    if ( v3 == 10 )
    {
      v4 = *(_QWORD *)(result + 88);
      if ( ((*(_BYTE *)(v4 + 193) & 0x10) != 0 || (*(_BYTE *)(v4 + 206) & 8) != 0) && *(_BYTE *)(v4 + 174) == 1 )
      {
        for ( i = *(_QWORD *)(v4 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( !**(_QWORD **)(i + 168) )
          return result;
      }
      if ( v2 != 17 )
        return 0;
    }
    else if ( v2 != 17 )
    {
      return 0;
    }
  }
  return result;
}
