// Function: sub_11FF460
// Address: 0x11ff460
//
__int64 __fastcall sub_11FF460(__int64 a1)
{
  int v1; // eax
  int v2; // eax
  unsigned __int64 v3; // rsi
  const char *v5; // [rsp+0h] [rbp-40h] BYREF
  char v6; // [rsp+20h] [rbp-20h]
  char v7; // [rsp+21h] [rbp-1Fh]

  while ( 1 )
  {
    do
    {
      v1 = sub_11FD3B0((unsigned __int8 **)a1);
      if ( v1 == -1 )
      {
LABEL_5:
        v3 = *(_QWORD *)(a1 + 56);
        v7 = 1;
        v6 = 3;
        v5 = "unterminated comment";
        sub_11FD800(a1, v3, (__int64)&v5, 2);
        return 1;
      }
    }
    while ( v1 != 42 );
    v2 = sub_11FD3B0((unsigned __int8 **)a1);
    if ( v2 == 47 )
      return 0;
    if ( v2 == -1 )
      goto LABEL_5;
  }
}
