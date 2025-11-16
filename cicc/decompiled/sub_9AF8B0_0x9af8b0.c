// Function: sub_9AF8B0
// Address: 0x9af8b0
//
__int64 __fastcall sub_9AF8B0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  __m128i v8; // [rsp+0h] [rbp-50h] BYREF
  __int64 v9; // [rsp+10h] [rbp-40h]
  __int64 v10; // [rsp+18h] [rbp-38h]
  __int64 v11; // [rsp+20h] [rbp-30h]
  __int64 v12; // [rsp+28h] [rbp-28h]
  __int64 v13; // [rsp+30h] [rbp-20h]
  __int64 v14; // [rsp+38h] [rbp-18h]
  char v15; // [rsp+40h] [rbp-10h]
  char v16; // [rsp+41h] [rbp-Fh]

  if ( !a5 || !*(_QWORD *)(a5 + 40) )
  {
    a5 = 0;
    if ( *(_BYTE *)a1 > 0x1Cu )
    {
      a5 = *(_QWORD *)(a1 + 40);
      if ( a5 )
        a5 = a1;
    }
  }
  v15 = a7;
  v8 = (__m128i)a2;
  v9 = 0;
  v10 = a6;
  v11 = a4;
  v12 = a5;
  v13 = 0;
  v14 = 0;
  v16 = 1;
  return sub_9AF7E0(a1, a3, &v8);
}
