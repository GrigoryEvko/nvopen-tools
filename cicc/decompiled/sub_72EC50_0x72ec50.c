// Function: sub_72EC50
// Address: 0x72ec50
//
__int64 *__fastcall sub_72EC50(__int64 a1)
{
  __int64 *result; // rax
  _BYTE *v2; // rdi

  result = (__int64 *)a1;
  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
    v2 = *(_BYTE **)(a1 + 56);
    if ( v2[173] == 12 && v2[176] == 1 && (v2[170] & 0x10) == 0 )
    {
      do
      {
        result = sub_72E9A0((__int64)v2);
        if ( *((_BYTE *)result + 24) != 2 )
          break;
        v2 = (_BYTE *)result[7];
        if ( v2[173] != 12 )
          break;
      }
      while ( v2[176] == 1 && (v2[170] & 0x10) == 0 );
    }
  }
  return result;
}
