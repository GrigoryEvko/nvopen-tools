// Function: sub_39A8220
// Address: 0x39a8220
//
__int64 __fastcall sub_39A8220(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  unsigned __int8 *v5; // r14
  __int64 v6; // rsi
  __int64 v7; // [rsp+8h] [rbp-28h]

  if ( a3 )
  {
    result = (__int64)sub_39A23D0(a1, (unsigned __int8 *)a2);
    v5 = (unsigned __int8 *)(a1 + 8);
    if ( result )
      return result;
  }
  else
  {
    v5 = sub_39A81B0(a1, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
    result = (__int64)sub_39A23D0(a1, (unsigned __int8 *)a2);
    if ( result )
      return result;
    v6 = *(_QWORD *)(a2 + 8 * (6LL - *(unsigned int *)(a2 + 8)));
    if ( v6 )
    {
      v5 = (unsigned __int8 *)(a1 + 8);
      sub_39A8220(a1, v6, 0);
    }
  }
  result = sub_39A5A90(a1, 46, (__int64)v5, (unsigned __int8 *)a2);
  if ( (*(_BYTE *)(a2 + 40) & 8) == 0 )
  {
    v7 = result;
    sub_39A70C0(a1, a2, result, 0);
    return v7;
  }
  return result;
}
