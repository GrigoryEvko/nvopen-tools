// Function: sub_22F50A0
// Address: 0x22f50a0
//
__int64 __fastcall sub_22F50A0(__int64 *a1, _BYTE *a2, size_t a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  size_t v6; // rdx

  if ( a3 == 1 && *a2 == 45 )
    return 1;
  v4 = *a1;
  v5 = *a1 + 16 * a1[1];
  if ( v5 == *a1 )
    return 1;
  while ( 1 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    if ( a3 >= v6 && (!v6 || !memcmp(a2, *(const void **)v4, v6)) )
      break;
    v4 += 16;
    if ( v5 == v4 )
      return 1;
  }
  return 0;
}
