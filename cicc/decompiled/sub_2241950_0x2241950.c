// Function: sub_2241950
// Address: 0x2241950
//
__int64 __fastcall sub_2241950(__int64 *a1, const void *a2, unsigned __int64 a3, size_t a4)
{
  __int64 v4; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // r12

  v4 = a1[1];
  if ( !v4 || !a4 )
    return -1;
  v6 = v4 - 1;
  v7 = *a1;
  if ( v6 <= a3 )
    a3 = v6;
  v8 = a3;
  do
  {
    if ( memchr(a2, *(char *)(v7 + v8), a4) )
      break;
  }
  while ( v8-- != 0 );
  return v8;
}
