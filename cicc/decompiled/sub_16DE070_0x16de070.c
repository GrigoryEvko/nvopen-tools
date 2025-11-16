// Function: sub_16DE070
// Address: 0x16de070
//
__int64 __fastcall sub_16DE070(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rdx

  v2 = sub_16F5C40();
  if ( v3 )
  {
    v6 = (unsigned int)sub_16F5EC0(v2, v3) - 6;
    result = 0;
    if ( (unsigned int)v6 <= 0x1A )
      return dword_42AF5E0[v6];
  }
  else
  {
    result = 0;
    if ( a2 > 7 )
    {
      v5 = (_QWORD *)(a1 + a2 - 8);
      if ( *v5 == 0x3361626D696C616BLL )
      {
        return 22;
      }
      else
      {
        result = 23;
        if ( *v5 != 0x3461626D696C616BLL )
        {
          result = 24;
          if ( *v5 != 0x3561626D696C616BLL )
            return 0;
        }
      }
    }
  }
  return result;
}
