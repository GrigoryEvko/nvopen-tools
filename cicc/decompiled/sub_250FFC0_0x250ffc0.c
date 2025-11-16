// Function: sub_250FFC0
// Address: 0x250ffc0
//
__int64 __fastcall sub_250FFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v8; // r12
  __int64 (__fastcall *v9)(__int64, void *); // r14
  void *v10; // rax

  v5 = *(__int64 **)(a1 + 40);
  result = *(unsigned int *)(a1 + 48);
  for ( i = &v5[result]; i != v5; result = v9(v8, v10) )
  {
    v8 = *v5++;
    v8 &= 0xFFFFFFFFFFFFFFF8LL;
    v9 = *(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 56LL);
    v10 = sub_CB7210(a1, a2, a3, a4, a5);
    a1 = v8;
    a2 = (__int64)v10;
  }
  return result;
}
