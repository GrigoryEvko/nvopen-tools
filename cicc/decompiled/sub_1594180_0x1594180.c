// Function: sub_1594180
// Address: 0x1594180
//
__int64 __fastcall sub_1594180(__int64 *a1)
{
  __int64 v1; // rax
  char v2; // dl

  v1 = *a1;
  v2 = *(_BYTE *)(*a1 + 8);
  if ( v2 == 14 || v2 == 16 )
    return *(unsigned int *)(v1 + 32);
  else
    return *(unsigned int *)(v1 + 12);
}
