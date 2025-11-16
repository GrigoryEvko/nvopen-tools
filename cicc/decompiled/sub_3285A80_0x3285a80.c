// Function: sub_3285A80
// Address: 0x3285a80
//
unsigned __int64 __fastcall sub_3285A80(unsigned __int16 *a1)
{
  unsigned __int16 v1; // ax
  __int64 v2; // rax

  v1 = *a1;
  if ( *a1 )
  {
    if ( v1 == 1 || (unsigned __int16)(v1 - 504) <= 7u )
      BUG();
    v2 = *(_QWORD *)&byte_444C4A0[16 * v1 - 16];
  }
  else
  {
    v2 = sub_3007260((__int64)a1);
  }
  return (v2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
}
