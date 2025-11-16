// Function: sub_17C2D30
// Address: 0x17c2d30
//
char *__fastcall sub_17C2D30(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  char *result; // rax
  unsigned __int64 *v4; // r8
  unsigned __int64 v5; // [rsp+8h] [rbp-8h] BYREF

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  result = *(char **)(v2 - 24);
  if ( result && (unsigned __int8)result[16] > 0x10u && (*(_BYTE *)(v2 + 16) != 78 || result[16] != 20) )
  {
    v5 = v2;
    v4 = *(unsigned __int64 **)(a1 + 8);
    if ( v4 == *(unsigned __int64 **)(a1 + 16) )
    {
      return sub_17C2330(a1, *(_BYTE **)(a1 + 8), &v5);
    }
    else
    {
      if ( v4 )
      {
        *v4 = v2;
        v4 = *(unsigned __int64 **)(a1 + 8);
      }
      *(_QWORD *)(a1 + 8) = v4 + 1;
    }
  }
  return result;
}
