// Function: sub_2EEFF00
// Address: 0x2eeff00
//
__int64 __fastcall sub_2EEFF00(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // rbx
  __int64 *i; // r13
  __int64 v5; // rsi
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = a2;
  result = (__int64)sub_2EEFC50(a1 + 600, v6);
  if ( !*(_BYTE *)result )
  {
    *(_BYTE *)result = 1;
    v3 = *(__int64 **)(v6[0] + 112LL);
    result = *(unsigned int *)(v6[0] + 120LL);
    for ( i = &v3[result]; i != v3; result = sub_2EEFF00(a1, v5) )
      v5 = *v3++;
  }
  return result;
}
