// Function: sub_1648C30
// Address: 0x1648c30
//
__int64 __fastcall sub_1648C30(__int64 a1, _QWORD *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx

  *a2 = 0;
  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 <= 0x17u )
  {
    if ( v2 == 18 )
    {
      v5 = *(_QWORD *)(a1 + 56);
      result = 0;
      if ( v5 )
LABEL_4:
        *a2 = *(_QWORD *)(v5 + 104);
    }
    else if ( v2 > 3u )
    {
      if ( v2 != 17 )
        return 1;
      v5 = *(_QWORD *)(a1 + 24);
      result = 0;
      if ( v5 )
        goto LABEL_4;
    }
    else
    {
      v6 = *(_QWORD *)(a1 + 40);
      result = 0;
      if ( v6 )
        *a2 = *(_QWORD *)(v6 + 120);
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 40);
    result = 0;
    if ( v3 )
    {
      v5 = *(_QWORD *)(v3 + 56);
      if ( v5 )
        goto LABEL_4;
    }
  }
  return result;
}
