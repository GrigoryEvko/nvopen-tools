// Function: sub_AF2BB0
// Address: 0xaf2bb0
//
unsigned __int64 __fastcall sub_AF2BB0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al

  if ( !a2 )
    return 0;
  v2 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 1 )
    return *(_QWORD *)(a2 + 136) & 0xFFFFFFFFFFFFFFF9LL;
  if ( (unsigned int)v2 - 25 <= 1 )
    return a2 & 0xFFFFFFFFFFFFFFF9LL | 2;
  if ( v2 == 7 )
    return a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
  else
    return 0;
}
