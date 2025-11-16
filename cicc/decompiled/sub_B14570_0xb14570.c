// Function: sub_B14570
// Address: 0xb14570
//
__int64 __fastcall sub_B14570(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, char a5)
{
  __int64 result; // rax
  __int64 *v6; // rdi
  unsigned __int64 v7; // r8
  __int64 v8; // rax

  result = (__int64)a2;
  if ( a2 != a3 )
  {
    do
    {
      *(_QWORD *)(result + 16) = a1;
      result = *(_QWORD *)(result + 8);
    }
    while ( a3 != (__int64 *)result );
  }
  if ( a5 )
    v6 = *(__int64 **)(a1 + 16);
  else
    v6 = (__int64 *)(a1 + 8);
  if ( v6 != a3 && a3 != a2 )
  {
    v7 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 8) = a3;
    *a3 = *a3 & 7 | *a2 & 0xFFFFFFFFFFFFFFF8LL;
    v8 = *v6;
    *(_QWORD *)(v7 + 8) = v6;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    *a2 = v8 | *a2 & 7;
    *(_QWORD *)(v8 + 8) = a2;
    result = v7 | *v6 & 7;
    *v6 = result;
  }
  return result;
}
