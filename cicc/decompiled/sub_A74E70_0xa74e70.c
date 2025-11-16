// Function: sub_A74E70
// Address: 0xa74e70
//
__int64 __fastcall sub_A74E70(__int64 a1, const void *a2, unsigned __int64 a3)
{
  __int64 *v3; // r12
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 *v6; // r14
  const void *v7; // rdi
  unsigned __int64 v8; // rdx
  size_t v9; // r8
  bool v10; // cc
  size_t v11; // rdx
  int v12; // eax
  size_t v15; // [rsp+10h] [rbp-50h]
  __int64 v17[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(__int64 **)(a1 + 8);
  if ( !*(_DWORD *)(a1 + 16) )
    return 0;
  v4 = *(unsigned int *)(a1 + 16);
  do
  {
    while ( 1 )
    {
      v5 = v4 >> 1;
      v6 = &v3[v4 >> 1];
      v17[0] = *v6;
      if ( !sub_A71840((__int64)v17) )
        goto LABEL_3;
      v7 = (const void *)sub_A71FD0(v17);
      v9 = v8;
      v10 = a3 <= v8;
      v11 = a3;
      if ( !v10 )
        v11 = v9;
      if ( v11 )
      {
        v15 = v9;
        v12 = memcmp(v7, a2, v11);
        v9 = v15;
        if ( v12 )
          break;
      }
      if ( a3 <= v9 )
        goto LABEL_10;
LABEL_3:
      v3 = v6 + 1;
      v4 = v4 - v5 - 1;
      if ( v4 <= 0 )
        goto LABEL_11;
    }
    if ( v12 < 0 )
      goto LABEL_3;
LABEL_10:
    v4 >>= 1;
  }
  while ( v5 > 0 );
LABEL_11:
  if ( v3 == (__int64 *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16))
    || !(unsigned __int8)sub_A721E0(v3, a2, a3) )
  {
    return 0;
  }
  return *v3;
}
