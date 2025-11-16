// Function: sub_8C3600
// Address: 0x8c3600
//
__int64 __fastcall sub_8C3600(__int64 a1, char a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 *v4; // rax

  v2 = a1;
  if ( a1 )
  {
    if ( (*(_BYTE *)(a1 - 8) & 2) != 0
      || (v3 = sub_72A270(a1, a2)) != 0
      && (v4 = *(__int64 **)(v3 + 32)) != 0
      && (a1 = *v4, (*(_BYTE *)(*v4 - 8) & 2) != 0) )
    {
      sub_8C3580(a1, a2);
    }
  }
  return v2;
}
