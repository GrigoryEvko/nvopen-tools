// Function: sub_11FE410
// Address: 0x11fe410
//
__int64 __fastcall sub_11FE410(__int64 a1, unsigned int a2)
{
  unsigned __int64 v2; // r13
  int v3; // eax
  unsigned __int64 v4; // rsi
  const char *v6; // [rsp+0h] [rbp-50h] BYREF
  char v7; // [rsp+20h] [rbp-30h]
  char v8; // [rsp+21h] [rbp-2Fh]

  v2 = *(_QWORD *)a1;
  do
  {
    v3 = sub_11FD3B0((unsigned __int8 **)a1);
    if ( v3 == -1 )
    {
      v4 = *(_QWORD *)(a1 + 56);
      v8 = 1;
      v7 = 3;
      v6 = "end of file in string constant";
      sub_11FD800(a1, v4, (__int64)&v6, 2);
      return 1;
    }
  }
  while ( v3 != 34 );
  sub_2241130(a1 + 72, 0, *(_QWORD *)(a1 + 80), v2, *(_QWORD *)a1 + ~v2);
  sub_11FCF00((_QWORD *)(a1 + 72));
  return a2;
}
