// Function: sub_5CF140
// Address: 0x5cf140
//
__int64 *__fastcall sub_5CF140(__int64 **a1, _QWORD *a2)
{
  __int64 *result; // rax
  __int64 *v3; // rdx

  result = *a1;
  if ( *a1 )
  {
    while ( 1 )
    {
      if ( (*((_BYTE *)result + 11) & 2) != 0 && (unsigned __int8)(*((_BYTE *)result + 10) - 2) > 1u )
      {
        *a2 = result;
        v3 = (__int64 *)*result;
        a2 = result;
        *result = 0;
        if ( !v3 )
          return result;
      }
      else
      {
        *a1 = result;
        v3 = (__int64 *)*result;
        a1 = (__int64 **)result;
        *result = 0;
        if ( !v3 )
          return result;
      }
      result = v3;
    }
  }
  return result;
}
