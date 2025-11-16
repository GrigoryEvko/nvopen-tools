// Function: sub_9374A0
// Address: 0x9374a0
//
__int64 __fastcall sub_9374A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  char v4; // al

  if ( *(char *)(a2 + 196) < 0 )
    sub_A77CD0(a1, 85);
  sub_A77B20(a1, 6);
  for ( result = *(_QWORD *)(a2 + 152); *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
    ;
  v3 = *(_QWORD *)(result + 168);
  if ( v3 )
  {
    v4 = *(_BYTE *)(v3 + 20);
    if ( (v4 & 8) != 0 )
    {
      sub_A77CD0(a1, 0);
      if ( (*(_BYTE *)(v3 + 20) & 1) == 0 )
        goto LABEL_8;
    }
    else if ( (v4 & 1) == 0 )
    {
LABEL_8:
      sub_A77B20(a1, 76);
      return sub_A77B20(a1, 19);
    }
    sub_A77B20(a1, 36);
    return sub_A77B20(a1, 19);
  }
  return result;
}
