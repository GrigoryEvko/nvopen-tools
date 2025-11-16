// Function: sub_23CCEB0
// Address: 0x23cceb0
//
void __fastcall sub_23CCEB0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // r14
  __int64 *v2; // r13
  unsigned __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // rbx
  void (__fastcall *v6)(__int64, __int64, __int64); // rax
  void (__fastcall *v7)(unsigned __int64, unsigned __int64, __int64); // rax
  void (__fastcall *v8)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  void (__fastcall *v13)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 i; // [rsp+8h] [rbp-58h]
  unsigned __int64 v16; // [rsp+18h] [rbp-48h]
  unsigned __int64 v17; // [rsp+20h] [rbp-40h]
  unsigned __int64 v18; // [rsp+28h] [rbp-38h]

  v1 = a1[7];
  v18 = a1[6];
  v2 = (__int64 *)(a1[5] + 8);
  v16 = a1[4];
  v3 = a1[2];
  v17 = a1[9];
  for ( i = a1[5]; v17 > (unsigned __int64)v2; ++v2 )
  {
    v4 = *v2;
    v5 = *v2 + 480;
    do
    {
      v6 = *(void (__fastcall **)(__int64, __int64, __int64))(v4 + 16);
      if ( v6 )
        v6(v4, v4, 3);
      v4 += 40;
    }
    while ( v5 != v4 );
  }
  if ( i != v17 )
  {
    for ( ; v16 != v3; v3 += 40LL )
    {
      v7 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v3 + 16);
      if ( v7 )
        v7(v3, v3, 3);
    }
    for ( ; v18 != v1; v1 += 40LL )
    {
      v8 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v1 + 16);
      if ( v8 )
        v8(v1, v1, 3);
    }
LABEL_16:
    v9 = *a1;
    if ( !*a1 )
      return;
    goto LABEL_17;
  }
  if ( v3 == v18 )
    goto LABEL_16;
  do
  {
    v13 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v3 + 16);
    if ( v13 )
      v13(v3, v3, 3);
    v3 += 40LL;
  }
  while ( v18 != v3 );
  v9 = *a1;
  if ( *a1 )
  {
LABEL_17:
    v10 = (unsigned __int64 *)a1[5];
    v11 = a1[9] + 8;
    if ( v11 > (unsigned __int64)v10 )
    {
      do
      {
        v12 = *v10++;
        j_j___libc_free_0(v12);
      }
      while ( v11 > (unsigned __int64)v10 );
      v9 = *a1;
    }
    j_j___libc_free_0(v9);
  }
}
