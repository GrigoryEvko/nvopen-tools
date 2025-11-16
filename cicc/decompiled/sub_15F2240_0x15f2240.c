// Function: sub_15F2240
// Address: 0x15f2240
//
__int64 __fastcall sub_15F2240(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  unsigned __int64 *v4; // r13
  __int64 v5; // r14
  unsigned __int64 v7; // rcx
  __int64 v8; // rax

  v4 = (unsigned __int64 *)a1[4];
  if ( v4 != (unsigned __int64 *)a3 )
  {
    v5 = (__int64)(a1 + 3);
    if ( a1 + 3 != a3 )
    {
      result = a1[5];
      if ( a2 + 40 != result + 40 )
        result = sub_157EA80(a2 + 40, result + 40, v5, (__int64)v4);
      if ( v4 != (unsigned __int64 *)a3 && (unsigned __int64 *)v5 != v4 )
      {
        v7 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 8) = v4;
        *v4 = *v4 & 7 | a1[3] & 0xFFFFFFFFFFFFFFF8LL;
        v8 = *a3;
        *(_QWORD *)(v7 + 8) = a3;
        v8 &= 0xFFFFFFFFFFFFFFF8LL;
        a1[3] = v8 | a1[3] & 7LL;
        *(_QWORD *)(v8 + 8) = v5;
        result = v7 | *a3 & 7;
        *a3 = result;
      }
    }
  }
  return result;
}
