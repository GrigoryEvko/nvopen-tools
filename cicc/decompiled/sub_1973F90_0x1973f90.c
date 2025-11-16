// Function: sub_1973F90
// Address: 0x1973f90
//
unsigned __int64 __fastcall sub_1973F90(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 *v4; // r15
  __int64 *v5; // r14
  __int64 v6; // r8
  __int64 v7; // rdi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rcx
  __int64 v10; // rax

  result = sub_157EBA0(a1);
  v4 = *(__int64 **)(a1 + 48);
  v5 = (__int64 *)(result + 24);
  if ( (__int64 *)(result + 24) != v4 )
  {
    v6 = a2 + 24;
    if ( v5 != (__int64 *)(a2 + 24) )
    {
      v7 = *(_QWORD *)(a2 + 40) + 40LL;
      v8 = result;
      if ( v7 != a1 + 40 )
      {
        result = sub_157EA80(v7, a1 + 40, *(_QWORD *)(a1 + 48), result + 24);
        v6 = a2 + 24;
      }
      if ( v5 != v4 )
      {
        v9 = *(_QWORD *)(v8 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v5;
        *(_QWORD *)(v8 + 24) = *(_QWORD *)(v8 + 24) & 7LL | *v4 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = *(_QWORD *)(a2 + 24);
        *(_QWORD *)(v9 + 8) = v6;
        v10 &= 0xFFFFFFFFFFFFFFF8LL;
        *v4 = v10 | *v4 & 7;
        *(_QWORD *)(v10 + 8) = v4;
        result = v9 | *(_QWORD *)(a2 + 24) & 7LL;
        *(_QWORD *)(a2 + 24) = result;
      }
    }
  }
  return result;
}
