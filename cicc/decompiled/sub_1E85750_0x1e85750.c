// Function: sub_1E85750
// Address: 0x1e85750
//
__int64 *__fastcall sub_1E85750(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 *v3; // rbx
  __int64 *i; // r13
  __int64 v5; // rsi
  __int64 *v6; // [rsp+8h] [rbp-28h] BYREF

  v6 = (__int64 *)a2;
  result = sub_1E855C0(a1 + 528, (__int64 *)&v6);
  if ( !*((_BYTE *)result + 8) )
  {
    *((_BYTE *)result + 8) = 1;
    result = v6;
    v3 = (__int64 *)v6[11];
    for ( i = (__int64 *)v6[12]; i != v3; result = (__int64 *)sub_1E85750(a1, v5) )
      v5 = *v3++;
  }
  return result;
}
