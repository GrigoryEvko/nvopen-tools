// Function: sub_82C0E0
// Address: 0x82c0e0
//
void __fastcall sub_82C0E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 i; // rbx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  v6 = qword_4F5F828;
  v7 = *(_QWORD *)(qword_4F5F828 + 3104);
  v8 = v7 + 1032 * (*(_QWORD *)(qword_4F5F828 + 3120) - 1LL);
  v9 = *(_QWORD *)(v8 + 1008);
  v10 = *(_QWORD *)(v8 + 1024);
  v18 = v8;
  v17 = v9;
  if ( v10 > 0 )
  {
    for ( i = 0; i != v10; ++i )
    {
      v12 = *(_QWORD *)(v9 + 32);
      if ( v12 )
      {
        sub_823A00(
          *(_QWORD *)v12,
          16LL * (unsigned int)(*(_DWORD *)(v12 + 8) + 1),
          v7,
          *(unsigned int *)(v12 + 8),
          a5,
          a6);
        sub_823A00(v12, 16, v13, v14, v15, v16);
      }
      v9 += 40;
    }
  }
  if ( v17 != v18 + 8 )
    sub_823A00(v17, 40LL * *(_QWORD *)(v18 + 1016), v18, a4, a5, a6);
  --*(_QWORD *)(v6 + 3120);
}
