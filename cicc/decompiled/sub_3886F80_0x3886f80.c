// Function: sub_3886F80
// Address: 0x3886f80
//
__int64 __fastcall sub_3886F80(__int64 a1)
{
  unsigned __int8 *v1; // r13
  unsigned __int8 *v2; // rax
  int v3; // eax
  const char *v4; // rax
  unsigned __int64 v5; // rsi
  __int64 result; // rax
  char v7; // r8
  signed __int64 v8; // rdx
  __int64 v9; // r13
  char *v10; // rax
  const char *v11; // [rsp+0h] [rbp-30h] BYREF
  char v12; // [rsp+10h] [rbp-20h]
  char v13; // [rsp+11h] [rbp-1Fh]

  v1 = *(unsigned __int8 **)(a1 + 48);
  v2 = sub_3880AC0(v1);
  if ( v2 )
  {
    *(_QWORD *)a1 = v2;
    sub_2241130((unsigned __int64 *)(a1 + 64), 0, *(_QWORD *)(a1 + 72), v1, v2 - 1 - v1);
    return 372;
  }
  if ( **(_BYTE **)a1 == 34 )
  {
    ++*(_QWORD *)a1;
    do
    {
      v3 = sub_3880F40((unsigned __int8 **)a1);
      if ( v3 == -1 )
      {
        v13 = 1;
        v4 = "end of file in COMDAT variable name";
LABEL_7:
        v5 = *(_QWORD *)(a1 + 48);
        v11 = v4;
        v12 = 3;
        sub_38814C0(a1, v5, (__int64)&v11);
        return 1;
      }
    }
    while ( v3 != 34 );
    sub_2241130(
      (unsigned __int64 *)(a1 + 64),
      0,
      *(_QWORD *)(a1 + 72),
      (_BYTE *)(*(_QWORD *)(a1 + 48) + 2LL),
      *(_QWORD *)a1 + ~(*(_QWORD *)(a1 + 48) + 2LL));
    sub_3880B30((unsigned __int64 *)(a1 + 64));
    v8 = *(_QWORD *)(a1 + 72);
    if ( v8 )
    {
      v9 = *(_QWORD *)(a1 + 64);
      if ( v8 < 0 )
        v8 = 0x7FFFFFFFFFFFFFFFLL;
      v10 = (char *)memchr(*(const void **)(a1 + 64), 0, v8);
      if ( v10 )
      {
        if ( &v10[-v9] != (char *)-1LL )
        {
          v13 = 1;
          v4 = "Null bytes are not allowed in names";
          goto LABEL_7;
        }
      }
    }
    return 374;
  }
  v7 = sub_3880FB0((unsigned __int64 *)a1);
  result = 1;
  if ( v7 )
    return 374;
  return result;
}
