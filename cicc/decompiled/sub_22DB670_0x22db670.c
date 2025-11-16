// Function: sub_22DB670
// Address: 0x22db670
//
__int64 __fastcall sub_22DB670(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // r14
  char v5; // al

  v1 = a1[4];
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 16);
    if ( !v2 )
      return 0;
    while ( 1 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v3 - 30) <= 0xAu )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 0;
    }
    v1 = 0;
LABEL_5:
    v4 = *(_QWORD *)(v3 + 40);
    v5 = sub_22DB400(a1, v4);
    if ( v4 && v5 )
    {
      if ( !v1 )
      {
        v1 = v4;
        goto LABEL_11;
      }
      return 0;
    }
LABEL_11:
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        break;
      v3 = *(_QWORD *)(v2 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v3 - 30) <= 0xAu )
        goto LABEL_5;
    }
  }
  return v1;
}
