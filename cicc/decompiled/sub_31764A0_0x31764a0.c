// Function: sub_31764A0
// Address: 0x31764a0
//
__int64 __fastcall sub_31764A0(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rsi

  if ( sub_B2FC80(a2) || !*(_QWORD *)(a2 + 104) || (unsigned __int8)sub_B2D610(a2, 27) )
    return 0;
  if ( !*(_BYTE *)(a1 + 180) )
  {
    if ( !sub_C8CA60(a1 + 152, a2) )
      goto LABEL_12;
    return 0;
  }
  v4 = *(_QWORD **)(a1 + 160);
  v5 = &v4[*(unsigned int *)(a1 + 172)];
  if ( v4 != v5 )
  {
    while ( a2 != *v4 )
    {
      if ( v5 == ++v4 )
        goto LABEL_12;
    }
    return 0;
  }
LABEL_12:
  if ( (unsigned __int8)sub_11F2A60(a2, 0, 0) )
    return 0;
  v6 = *(_QWORD *)(a2 + 80);
  if ( v6 )
    v6 -= 24;
  if ( !(unsigned __int8)sub_2A64220(*(__int64 **)a1, v6) )
    return 0;
  return (unsigned int)sub_B2D610(a2, 3) ^ 1;
}
