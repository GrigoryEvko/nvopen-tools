// Function: sub_266EDE0
// Address: 0x266ede0
//
__int64 __fastcall sub_266EDE0(__int64 *a1, __int64 a2, __int64 *a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rdx

  v3 = 0;
  if ( !*(_BYTE *)(*a1 + 241) )
  {
    v4 = *a3;
    v3 = 1;
    if ( v4 )
      sub_250ED80(a2, *a1, v4, 1);
  }
  return v3;
}
