// Function: sub_11FF310
// Address: 0x11ff310
//
__int64 __fastcall sub_11FF310(__int64 a1)
{
  unsigned __int8 *v1; // r13
  unsigned __int8 *v2; // rax
  int v3; // eax
  const char *v4; // rax
  unsigned __int64 v5; // rsi
  size_t v7; // rdx
  const char *v8; // [rsp+0h] [rbp-40h] BYREF
  char v9; // [rsp+20h] [rbp-20h]
  char v10; // [rsp+21h] [rbp-1Fh]

  v1 = *(unsigned __int8 **)(a1 + 56);
  v2 = sub_11FD0C0(v1);
  if ( v2 )
  {
    *(_QWORD *)a1 = v2;
    sub_2241130(a1 + 72, 0, *(_QWORD *)(a1 + 80), v1, v2 - 1 - v1);
    return 507;
  }
  else if ( **(_BYTE **)a1 == 34 )
  {
    ++*(_QWORD *)a1;
    do
    {
      v3 = sub_11FD3B0((unsigned __int8 **)a1);
      if ( v3 == -1 )
      {
        v10 = 1;
        v4 = "end of file in COMDAT variable name";
LABEL_7:
        v5 = *(_QWORD *)(a1 + 56);
        v8 = v4;
        v9 = 3;
        sub_11FD800(a1, v5, (__int64)&v8, 2);
        return 1;
      }
    }
    while ( v3 != 34 );
    sub_2241130(
      a1 + 72,
      0,
      *(_QWORD *)(a1 + 80),
      *(_QWORD *)(a1 + 56) + 2LL,
      *(_QWORD *)a1 + ~(*(_QWORD *)(a1 + 56) + 2LL));
    sub_11FCF00((_QWORD *)(a1 + 72));
    v7 = *(_QWORD *)(a1 + 80);
    if ( v7 && memchr(*(const void **)(a1 + 72), 0, v7) )
    {
      v10 = 1;
      v4 = "NUL character is not allowed in names";
      goto LABEL_7;
    }
    return 509;
  }
  else
  {
    return (unsigned __int8)sub_11FD420((unsigned __int8 **)a1) == 0 ? 1 : 509;
  }
}
