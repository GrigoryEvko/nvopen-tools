// Function: sub_14594D0
// Address: 0x14594d0
//
__int64 __fastcall sub_14594D0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r12

  if ( (a1[5] & 0xFFFFFFFFFFFFFFF8LL) == 0
    || (v4 = a1[5] & 0xFFFFFFFFFFFFFFF8LL, sub_1456E90(a3) == v4)
    || !(unsigned __int8)sub_14594A0(a3, a1[5] & 0xFFFFFFFFFFFFFFF8LL, a2) )
  {
    v5 = *a1;
    v6 = *a1 + 24LL * *((unsigned int *)a1 + 2);
    if ( v6 == *a1 )
      return 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(v5 + 8);
      if ( v7 != sub_1456E90(a3) )
      {
        if ( (unsigned __int8)sub_14594A0(a3, *(_QWORD *)(v5 + 8), a2) )
          break;
      }
      v5 += 24;
      if ( v6 == v5 )
        return 0;
    }
  }
  return 1;
}
