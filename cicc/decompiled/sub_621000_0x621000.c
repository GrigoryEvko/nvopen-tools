// Function: sub_621000
// Address: 0x621000
//
__int64 __fastcall sub_621000(__int16 *a1, int a2, __int16 *a3, int a4)
{
  __int64 v5; // rax
  unsigned __int16 v6; // si

  if ( !a2 || *a1 >= 0 )
  {
    if ( a4 && *a3 < 0 )
      return 1;
LABEL_8:
    v5 = 0;
    while ( 1 )
    {
      v6 = a3[v5];
      if ( (unsigned __int16)a1[v5] > v6 )
        break;
      if ( (unsigned __int16)a1[v5] < v6 )
        return 0xFFFFFFFFLL;
      if ( ++v5 == 8 )
        return 0;
    }
    return 1;
  }
  if ( a4 && *a3 < 0 )
    goto LABEL_8;
  return 0xFFFFFFFFLL;
}
