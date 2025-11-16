// Function: sub_14AA530
// Address: 0x14aa530
//
__int64 __fastcall sub_14AA530(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  int v4; // edx
  __int64 v5; // rdi

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 1;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    if ( *(_BYTE *)(v2 + 16) != 75 )
      break;
    v4 = *(unsigned __int16 *)(v2 + 18);
    BYTE1(v4) &= ~0x80u;
    if ( (unsigned int)(v4 - 32) > 1 )
      break;
    v5 = *(_QWORD *)(v2 - 24);
    if ( *(_BYTE *)(v5 + 16) > 0x10u || !(unsigned __int8)sub_1593BB0(v5) )
      break;
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 1;
  }
  return 0;
}
