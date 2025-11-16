// Function: sub_9B64A0
// Address: 0x9b64a0
//
__int64 __fastcall sub_9B64A0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  _QWORD v9[8]; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+40h] [rbp-10h]
  char v11; // [rsp+41h] [rbp-Fh]

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
  v9[0] = a2;
  v10 = a8;
  v9[1] = 0;
  v9[2] = 0;
  v9[3] = a7;
  v9[4] = a5;
  v9[5] = a6;
  v9[6] = 0;
  v9[7] = 0;
  v11 = 1;
  return sub_9A1DB0((unsigned __int8 *)a1, a3, a4, (__int64)v9, a5);
}
