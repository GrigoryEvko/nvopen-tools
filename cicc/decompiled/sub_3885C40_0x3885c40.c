// Function: sub_3885C40
// Address: 0x3885c40
//
__int64 __fastcall sub_3885C40(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // r13
  int v3; // eax
  unsigned __int64 v4; // rsi
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+10h] [rbp-30h]
  char v8; // [rsp+11h] [rbp-2Fh]

  v2 = *(_QWORD *)a1;
  do
  {
    v3 = sub_3880F40((unsigned __int8 **)a1);
    if ( v3 == -1 )
    {
      v4 = *(_QWORD *)(a1 + 48);
      v8 = 1;
      v7 = 3;
      v6 = "end of file in string constant";
      sub_38814C0(a1, v4, (__int64)&v6);
      return 1;
    }
  }
  while ( v3 != 34 );
  sub_2241130((unsigned __int64 *)(a1 + 64), 0, *(_QWORD *)(a1 + 72), (_BYTE *)v2, *(_QWORD *)a1 + ~v2);
  sub_3880B30((unsigned __int64 *)(a1 + 64));
  return a2;
}
