// Function: sub_1196BB0
// Address: 0x1196bb0
//
char __fastcall sub_1196BB0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  char result; // al
  unsigned __int8 *v4; // rdx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 != 58 )
    return 0;
  result = sub_10A7530(a1, 15, *(unsigned __int8 **)(a2 - 64));
  v4 = *(unsigned __int8 **)(a2 - 32);
  if ( !result || v4 != (unsigned __int8 *)*a1[2] )
  {
    if ( (unsigned __int8)sub_10A7530(a1, 15, v4) )
      return *a1[2] == *(_QWORD *)(a2 - 64);
    return 0;
  }
  return result;
}
