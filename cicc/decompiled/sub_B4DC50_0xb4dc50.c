// Function: sub_B4DC50
// Address: 0xb4dc50
//
__int64 __fastcall sub_B4DC50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 *v4; // rbx

  if ( !a3 )
    return a1;
  v3 = (__int64 *)(a2 + 8 * a3);
  v4 = (__int64 *)(a2 + 8);
  if ( (__int64 *)(a2 + 8) != v3 )
  {
    do
    {
      a1 = sub_B4DB80(a1, *v4);
      if ( !a1 )
        break;
      ++v4;
    }
    while ( v3 != v4 );
  }
  return a1;
}
