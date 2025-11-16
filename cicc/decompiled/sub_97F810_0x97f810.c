// Function: sub_97F810
// Address: 0x97f810
//
__int64 __fastcall sub_97F810(__int64 *a1, _QWORD *a2, unsigned __int8 (__fastcall *a3)(__int64, _QWORD, _QWORD))
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15

  v3 = *a1;
  v4 = a1[1] - *a1;
  v5 = v4 >> 6;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5 >> 1;
        if ( !a3(v3 + (v5 >> 1 << 6), *a2, a2[1]) )
          break;
        v3 += (v5 >> 1 << 6) + 64;
        v5 = v5 - v6 - 1;
        if ( v5 <= 0 )
          return v3;
      }
      v5 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v3;
}
