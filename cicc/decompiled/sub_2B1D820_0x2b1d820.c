// Function: sub_2B1D820
// Address: 0x2b1d820
//
__int64 __fastcall sub_2B1D820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // r12

  v6 = a2 - a1;
  v8 = a1;
  v9 = v6 >> 6;
  if ( v6 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v10 = v9 >> 1;
        v11 = v8 + (v9 >> 1 << 6);
        if ( (unsigned __int8)sub_2B1D420(
                                *(unsigned __int8 **)(*(_QWORD *)a3 + 8LL),
                                *(unsigned __int8 **)(*(_QWORD *)v11 + 8LL),
                                a3,
                                a4,
                                a5,
                                a6) )
          break;
        v8 = v11 + 64;
        v9 = v9 - v10 - 1;
        if ( v9 <= 0 )
          return v8;
      }
      v9 >>= 1;
    }
    while ( v10 > 0 );
  }
  return v8;
}
