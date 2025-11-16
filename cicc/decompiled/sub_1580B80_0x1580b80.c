// Function: sub_1580B80
// Address: 0x1580b80
//
__int64 __fastcall sub_1580B80(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 *v3; // r13
  unsigned __int64 *v4; // r12
  __int64 v5; // r14
  __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  __int64 v9; // rax

  v3 = (unsigned __int64 *)(a2 + 24);
  v4 = (unsigned __int64 *)a1[4];
  if ( (unsigned __int64 *)(a2 + 24) != v4 )
  {
    v5 = (__int64)(a1 + 3);
    if ( a1 + 3 != v3 )
    {
      v7 = *(_QWORD *)(a2 + 56) + 72LL;
      result = a1[7];
      if ( v7 != result + 72 )
        result = sub_15809C0(v7, result + 72, v5, (__int64)v4);
      if ( v3 != v4 && (unsigned __int64 *)v5 != v4 )
      {
        v8 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 8) = v4;
        *v4 = *v4 & 7 | a1[3] & 0xFFFFFFFFFFFFFFF8LL;
        v9 = *(_QWORD *)(a2 + 24);
        *(_QWORD *)(v8 + 8) = v3;
        v9 &= 0xFFFFFFFFFFFFFFF8LL;
        a1[3] = v9 | a1[3] & 7LL;
        *(_QWORD *)(v9 + 8) = v5;
        result = v8 | *(_QWORD *)(a2 + 24) & 7LL;
        *(_QWORD *)(a2 + 24) = result;
      }
    }
  }
  return result;
}
