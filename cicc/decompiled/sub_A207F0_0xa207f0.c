// Function: sub_A207F0
// Address: 0xa207f0
//
__int64 __fastcall sub_A207F0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi

  result = a1[4];
  if ( a1[3] != result && *(_QWORD *)(result - 32) == a2 )
  {
    sub_A19830(*a1, 0x12u, 3u);
    for ( i = a1[4]; i != a1[3]; i = a1[4] )
    {
      if ( *(_QWORD *)(i - 32) != a2 )
        break;
      sub_A20640(a1, i - 40);
      v6 = a1[4];
      a1[4] = v6 - 40;
      v7 = *(_QWORD *)(v6 - 24);
      if ( v7 )
        j_j___libc_free_0(v7, *(_QWORD *)(v6 - 8) - v7);
    }
    return sub_A192A0(*a1);
  }
  return result;
}
