// Function: sub_39A3B30
// Address: 0x39a3b30
//
__int64 __fastcall sub_39A3B30(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  int v3; // r14d
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 i; // [rsp+8h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 392);
  result = (__int64)&v1[3 * *(unsigned int *)(a1 + 400)];
  for ( i = result; (__int64 *)i != v1; v1 += 3 )
  {
    v3 = *((_DWORD *)v1 + 4);
    v4 = *v1;
    v5 = v1[1];
    if ( (v3 & 1) != 0 )
    {
      result = (__int64)sub_39A23D0(a1, *(unsigned __int8 **)(v4 - 8LL * *(unsigned int *)(v4 + 8)));
      if ( result )
        result = sub_39A3B20(a1, v5, 34, result);
    }
    if ( (v3 & 2) != 0 )
    {
      result = (__int64)sub_39A23D0(a1, *(unsigned __int8 **)(v4 + 8 * (2LL - *(unsigned int *)(v4 + 8))));
      if ( result )
        result = sub_39A3B20(a1, v5, 47, result);
    }
  }
  *(_DWORD *)(a1 + 400) = 0;
  return result;
}
