// Function: sub_266EE20
// Address: 0x266ee20
//
__int64 __fastcall sub_266EE20(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // r12d

  v5 = *a1;
  v6 = *a3;
  v7 = *(unsigned __int8 *)(*a1 + 241);
  if ( !(_BYTE)v7 )
  {
    if ( *(_BYTE *)(v5 + 113) )
      return v7;
    v7 = 1;
  }
  if ( v6 )
    sub_250ED80(a2, v5, v6, 1);
  return v7;
}
