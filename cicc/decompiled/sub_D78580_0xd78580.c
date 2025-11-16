// Function: sub_D78580
// Address: 0xd78580
//
bool __fastcall sub_D78580(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r12
  unsigned __int8 *v5; // rbx
  bool result; // al

  if ( !a1 )
    return 0;
  if ( sub_B46AA0(a1) )
    return 0;
  v3 = *(_QWORD *)(a1 - 32);
  v4 = 0;
  if ( *(_BYTE *)a1 == 85 )
    v4 = a1;
  if ( !v3 )
    BUG();
  if ( *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 80) )
  {
    v5 = sub_BD3990((unsigned __int8 *)v3, a2);
    v3 = (__int64)v5;
    if ( *v5 )
    {
      if ( *v5 != 1 || (v3 = sub_B325F0((__int64)v5), *(_BYTE *)v3) )
      {
        if ( (_BYTE)qword_4F878A8 && (!v4 || **(_BYTE **)(v4 - 32) != 25) )
          return *v5 > 0x15u;
        return 0;
      }
    }
  }
  result = 1;
  if ( v4 )
    return ((*(_BYTE *)(v3 + 33) >> 5) ^ 1) & 1;
  return result;
}
