// Function: sub_8C3810
// Address: 0x8c3810
//
__int64 *__fastcall sub_8C3810(__int64 *a1, unsigned __int8 a2)
{
  __int64 *result; // rax
  char v3; // dl
  char v5; // si
  __int64 *v6; // [rsp-10h] [rbp-10h]

  result = a1;
  if ( a1 )
  {
    v3 = *((_BYTE *)a1 - 8);
    v5 = a2 - 6;
    if ( (v3 & 3) == 3 )
    {
      while ( (v3 & 8) != 0 )
      {
        switch ( v5 )
        {
          case 0:
          case 1:
          case 5:
          case 22:
          case 53:
            result = (__int64 *)result[14];
            break;
          case 16:
            result = (__int64 *)result[7];
            break;
          case 17:
            result = (__int64 *)*result;
            break;
          default:
            sub_721090();
        }
        if ( result )
        {
          v3 = *((_BYTE *)result - 8);
          if ( (v3 & 3) == 3 )
            continue;
        }
        return result;
      }
      v6 = result;
      sub_8C3650(result, a2, dword_4F60238 == 0);
      result = (__int64 *)*(v6 - 3);
      if ( (*(_BYTE *)(result - 1) & 2) != 0 )
        return (__int64 *)*(result - 3);
    }
  }
  return result;
}
