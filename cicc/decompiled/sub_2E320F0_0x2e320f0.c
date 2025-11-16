// Function: sub_2E320F0
// Address: 0x2e320f0
//
unsigned __int64 *__fastcall sub_2E320F0(__int64 *a1, __int64 a2)
{
  unsigned __int64 *result; // rax
  unsigned __int64 *v3; // rdx
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rdx

  result = *(unsigned __int64 **)(a2 + 8);
  v3 = (unsigned __int64 *)a1[1];
  if ( result != v3 && a1 != (__int64 *)result && v3 != result && a1 != (__int64 *)v3 )
  {
    v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v3;
    *v3 = *v3 & 7 | *a1 & 0xFFFFFFFFFFFFFFF8LL;
    v5 = *result;
    *(_QWORD *)(v4 + 8) = result;
    v5 &= 0xFFFFFFFFFFFFFFF8LL;
    *a1 = v5 | *a1 & 7;
    *(_QWORD *)(v5 + 8) = a1;
    *result = v4 | *result & 7;
  }
  return result;
}
