// Function: sub_2210710
// Address: 0x2210710
//
__int64 __fastcall sub_2210710(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4, char a5)
{
  unsigned __int64 v7; // rbp
  unsigned __int64 v8; // rbx
  unsigned int v9; // eax
  _QWORD v11[6]; // [rsp+0h] [rbp-30h] BYREF

  v11[0] = a1;
  v11[1] = a2;
  if ( (a5 & 4) != 0 && (unsigned __int64)(a2 - a1) > 2 && *(_WORD *)a1 == 0xBBEF && *(_BYTE *)(a1 + 2) == 0xBF )
    v11[0] = a1 + 3;
  if ( a3 > 1 )
  {
    v7 = 0;
    v8 = 1;
    while ( 1 )
    {
      v9 = sub_220F920((__int64)v11, a4);
      if ( a4 < v9 )
        return v11[0];
      if ( v9 <= 0xFFFF )
        v8 = v7;
      v7 = v8 + 1;
      v8 += 2LL;
      if ( v8 >= a3 )
        goto LABEL_10;
    }
  }
  v8 = 1;
LABEL_10:
  if ( a3 != v8 )
    return v11[0];
  if ( a4 >= 0xFFFF )
    a4 = 0xFFFF;
  sub_220F920((__int64)v11, a4);
  return v11[0];
}
