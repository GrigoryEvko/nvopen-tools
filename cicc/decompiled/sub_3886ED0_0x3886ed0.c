// Function: sub_3886ED0
// Address: 0x3886ed0
//
__int64 __fastcall sub_3886ED0(__int64 a1)
{
  __int64 result; // rax
  signed __int64 v2; // rdx
  __int64 v3; // rbx
  char *v4; // rax
  unsigned __int64 v5; // rsi
  const char *v6; // [rsp+0h] [rbp-30h] BYREF
  char v7; // [rsp+10h] [rbp-20h]
  char v8; // [rsp+11h] [rbp-1Fh]

  result = sub_3885C40(a1, 0x179u);
  if ( (unsigned int)result > 1 && **(_BYTE **)a1 == 58 )
  {
    ++*(_QWORD *)a1;
    v2 = *(_QWORD *)(a1 + 72);
    if ( !v2 )
      return 372;
    v3 = *(_QWORD *)(a1 + 64);
    if ( v2 < 0 )
      v2 = 0x7FFFFFFFFFFFFFFFLL;
    v4 = (char *)memchr(*(const void **)(a1 + 64), 0, v2);
    if ( v4 && &v4[-v3] != (char *)-1LL )
    {
      v5 = *(_QWORD *)(a1 + 48);
      v8 = 1;
      v7 = 3;
      v6 = "Null bytes are not allowed in names";
      sub_38814C0(a1, v5, (__int64)&v6);
      return 1;
    }
    else
    {
      return 372;
    }
  }
  return result;
}
