// Function: sub_14438E0
// Address: 0x14438e0
//
__int64 __fastcall sub_14438E0(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r14

  v1 = a1[4];
  if ( !v1 )
    return 0;
  v2 = *(_QWORD *)(v1 + 8);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v3 = sub_1648700(v2);
    if ( (unsigned __int8)(*(_BYTE *)(v3 + 16) - 25) <= 9u )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  v4 = 0;
LABEL_6:
  v5 = *(_QWORD *)(v3 + 40);
  if ( !(unsigned __int8)sub_1443560(a1, v5) )
    goto LABEL_9;
  if ( v4 )
    return 0;
  v4 = v5;
LABEL_9:
  while ( 1 )
  {
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return v4;
    v3 = sub_1648700(v2);
    if ( (unsigned __int8)(*(_BYTE *)(v3 + 16) - 25) <= 9u )
      goto LABEL_6;
  }
}
