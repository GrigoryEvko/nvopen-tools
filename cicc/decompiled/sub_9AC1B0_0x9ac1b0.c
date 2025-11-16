// Function: sub_9AC1B0
// Address: 0x9ac1b0
//
__int64 __fastcall sub_9AC1B0(
        __int64 a1,
        unsigned __int64 *a2,
        unsigned __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  __m128i v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 v11; // [rsp+18h] [rbp-38h]
  __int64 v12; // [rsp+20h] [rbp-30h]
  __int64 v13; // [rsp+28h] [rbp-28h]
  __int64 v14; // [rsp+30h] [rbp-20h]
  __int64 v15; // [rsp+38h] [rbp-18h]
  char v16; // [rsp+40h] [rbp-10h]
  char v17; // [rsp+41h] [rbp-Fh]

  if ( !a6 || !*(_QWORD *)(a6 + 40) )
  {
    a6 = 0;
    if ( *(_BYTE *)a1 > 0x1Cu )
    {
      a6 = *(_QWORD *)(a1 + 40);
      if ( a6 )
        a6 = a1;
    }
  }
  v9 = (__m128i)a3;
  v16 = a8;
  v10 = 0;
  v11 = a7;
  v12 = a5;
  v13 = a6;
  v14 = 0;
  v15 = 0;
  v17 = 1;
  return sub_9AC0E0(a1, a2, a4, &v9);
}
