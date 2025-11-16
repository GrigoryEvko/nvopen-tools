// Function: sub_1227E40
// Address: 0x1227e40
//
__int64 __fastcall sub_1227E40(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // rbp
  __int64 v5; // rdi
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp-38h] [rbp-38h] BYREF
  char v8; // [rsp-18h] [rbp-18h]
  char v9; // [rsp-17h] [rbp-17h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( a3 )
    return sub_12273B0(a1, a2);
  v10 = v3;
  v5 = a1 + 176;
  v6 = *(_QWORD *)(v5 + 56);
  v7 = "missing 'distinct', required for !DICompileUnit";
  v9 = 1;
  v8 = 3;
  sub_11FD800(v5, v6, (__int64)&v7, 1);
  return 1;
}
