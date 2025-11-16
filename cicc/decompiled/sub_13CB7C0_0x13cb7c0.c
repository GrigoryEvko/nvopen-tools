// Function: sub_13CB7C0
// Address: 0x13cb7c0
//
__int64 __fastcall sub_13CB7C0(_BYTE *a1)
{
  int v1; // r14d
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // r12

  if ( a1[16] > 0x10u )
    return 0;
  if ( (unsigned __int8)sub_1593BB0(a1) )
    return 1;
  if ( a1[16] == 9 )
    return 1;
  v1 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !v1 )
    return 1;
  v2 = 0;
  while ( 1 )
  {
    v3 = sub_15A0A60(a1, v2);
    v4 = v3;
    if ( !v3 || !(unsigned __int8)sub_1593BB0(v3) && *(_BYTE *)(v4 + 16) != 9 )
      break;
    if ( v1 == ++v2 )
      return 1;
  }
  return 0;
}
