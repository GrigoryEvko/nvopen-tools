// Function: sub_1462D40
// Address: 0x1462d40
//
void __fastcall sub_1462D40(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 *v10; // r15
  __int64 **v11; // r14
  __int64 *v12; // r15
  __int64 v13; // r8
  __int64 **v14; // [rsp+18h] [rbp-48h]
  __int64 **v15; // [rsp+20h] [rbp-40h]

  if ( a1 != (__int64 *)a2 && a2 != (__int64 **)(a1 + 1) )
  {
    v14 = (__int64 **)(a1 + 1);
    do
    {
      while ( (int)sub_1462150(a7, a8, *a9, *v14, *a1, a10, 0) >= 0 )
      {
        v11 = v14;
        v12 = *v14;
        while ( 1 )
        {
          v13 = (__int64)*(v11 - 1);
          v15 = v11--;
          if ( (int)sub_1462150(a7, a8, *a9, v12, v13, a10, 0) >= 0 )
            break;
          v11[1] = *v11;
        }
        *v15 = v12;
        if ( a2 == ++v14 )
          return;
      }
      v10 = *v14;
      if ( a1 != (__int64 *)v14 )
        memmove(a1 + 1, a1, (char *)v14 - (char *)a1);
      *a1 = (__int64)v10;
      ++v14;
    }
    while ( a2 != v14 );
  }
}
