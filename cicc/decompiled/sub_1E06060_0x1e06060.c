// Function: sub_1E06060
// Address: 0x1e06060
//
// bad sp value at call has been detected, the output may be wrong!
void __fastcall sub_1E06060(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  char *v8[6]; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 != a2 )
  {
    v8[0] = (char *)a1;
    v3 = sub_1E048B0(*(_QWORD **)(v2 + 24), *(_QWORD *)(v2 + 32), (__int64 *)v8);
    sub_1D82C50(*(_QWORD *)(a1 + 8) + 24LL, v3);
    v8[0] = (char *)a1;
    *(_QWORD *)(a1 + 8) = a2;
    sub_1E06030(a2 + 24, v8);
    sub_1E05100(a1, (__int64)v8, v4, v5, v6, v7);
  }
}
