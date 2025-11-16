// Function: sub_2B099C0
// Address: 0x2b099c0
//
__int64 __fastcall sub_2B099C0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx

  result = 1;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    if ( (unsigned __int8)sub_B46420(a1) )
      return 0;
    if ( (unsigned __int8)sub_B46490(a1) )
      return 0;
    result = sub_BD3660(a1, 64);
    if ( (_BYTE)result )
    {
      return 0;
    }
    else
    {
      v2 = *(_QWORD *)(a1 + 16);
      if ( v2 )
      {
        while ( 1 )
        {
          v3 = *(_QWORD *)(v2 + 24);
          if ( *(_BYTE *)v3 > 0x1Cu && *(_QWORD *)(v3 + 40) == *(_QWORD *)(a1 + 40) && *(_BYTE *)v3 != 84 )
            break;
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            return 1;
        }
      }
      else
      {
        return 1;
      }
    }
  }
  return result;
}
