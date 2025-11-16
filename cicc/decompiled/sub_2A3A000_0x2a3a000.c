// Function: sub_2A3A000
// Address: 0x2a3a000
//
unsigned __int64 __fastcall sub_2A3A000(__int64 a1)
{
  __int64 v1; // r8
  unsigned __int64 result; // rax

  if ( *(_BYTE *)a1 != 30 )
  {
    v1 = 0;
    if ( ((*(_BYTE *)a1 - 35) & 0xFD) != 0 )
      return v1;
    return a1;
  }
  result = sub_AA4E50(*(_QWORD *)(a1 + 40));
  if ( !result )
    return a1;
  return result;
}
