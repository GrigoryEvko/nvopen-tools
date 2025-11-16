// Function: sub_623FC0
// Address: 0x623fc0
//
__int64 __fastcall sub_623FC0(__int64 a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 i; // rax
  __int64 v8; // rax

  *a3 = 0;
  *a2 = 0;
  *a4 = 0;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    return 0;
  if ( !(unsigned int)sub_8D2310(a1) )
    return 0;
  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  *a2 = i;
  v8 = *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL);
  if ( !v8 )
    return 0;
  *a3 = v8;
  *a4 = *(_QWORD *)a1;
  return 1;
}
