// Function: sub_1C9A550
// Address: 0x1c9a550
//
char *__fastcall sub_1C9A550(__int64 a1, __int64 a2)
{
  __int64 *v3; // rsi
  char *result; // rax
  _BYTE *v5; // rsi
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(__int64 **)(a2 - 48);
  result = (char *)*v3;
  if ( *(_BYTE *)(*v3 + 8) == 13 )
  {
    result = (char *)sub_1C9A0C0(*(_QWORD *)(a2 - 24), v3, *(_BYTE *)(a2 + 18) & 1, a2);
    v6[0] = a2;
    v5 = *(_BYTE **)(a1 + 432);
    if ( v5 == *(_BYTE **)(a1 + 440) )
    {
      return sub_17C2330(a1 + 424, v5, v6);
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = a2;
        v5 = *(_BYTE **)(a1 + 432);
      }
      *(_QWORD *)(a1 + 432) = v5 + 8;
    }
  }
  return result;
}
