// Function: sub_157F120
// Address: 0x157f120
//
__int64 __fastcall sub_157F120(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 + 8);
  do
  {
    if ( !v1 )
      return 0;
    v2 = sub_1648700(v1);
    v1 = *(_QWORD *)(v1 + 8);
  }
  while ( (unsigned __int8)(*(_BYTE *)(v2 + 16) - 25) > 9u );
  v3 = *(_QWORD *)(v2 + 40);
  if ( !v1 )
    return v3;
  while ( 1 )
  {
    v4 = sub_1648700(v1);
    if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 25) <= 9u )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v3;
  }
LABEL_9:
  if ( v3 != *(_QWORD *)(v4 + 40) )
    return 0;
  while ( 1 )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v3;
    v4 = sub_1648700(v1);
    if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 25) <= 9u )
      goto LABEL_9;
  }
}
