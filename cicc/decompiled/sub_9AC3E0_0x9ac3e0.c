// Function: sub_9AC3E0
// Address: 0x9ac3e0
//
__int64 __fastcall sub_9AC3E0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  __m128i v9; // [rsp+0h] [rbp-60h] BYREF
  __int64 v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]
  __int64 v13; // [rsp+28h] [rbp-38h]
  __int64 v14; // [rsp+30h] [rbp-30h]
  __int64 v15; // [rsp+38h] [rbp-28h]
  char v16; // [rsp+40h] [rbp-20h]
  char v17; // [rsp+41h] [rbp-1Fh]

  if ( !a6 || !*(_QWORD *)(a6 + 40) )
  {
    a6 = 0;
    if ( *(_BYTE *)a2 > 0x1Cu )
    {
      a6 = *(_QWORD *)(a2 + 40);
      if ( a6 )
        a6 = a2;
    }
  }
  v9 = (__m128i)a3;
  v16 = a8;
  v11 = a7;
  v10 = 0;
  v12 = a5;
  v13 = a6;
  v14 = 0;
  v15 = 0;
  v17 = 1;
  sub_9AC330(a1, a2, a4, &v9);
  return a1;
}
