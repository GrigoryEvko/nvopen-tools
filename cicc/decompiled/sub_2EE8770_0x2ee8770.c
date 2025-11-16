// Function: sub_2EE8770
// Address: 0x2ee8770
//
__int64 __fastcall sub_2EE8770(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  int *v7; // rax
  __int64 *v8; // rdx
  __int64 *v9; // r13
  __int64 *v10; // r15
  __int64 v11; // r12
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned int v14; // ecx
  int v16; // [rsp+8h] [rbp-38h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

  if ( !*(_DWORD *)(a2 + 72) )
    return 0;
  v2 = sub_2EE86D0(a1, a2);
  if ( v2 )
  {
    if ( a2 == **(_QWORD **)(v2 + 32) )
      return 0;
  }
  v7 = (int *)sub_2EE8230(*(_QWORD *)(a1 + 440), a2, v3, v4, v5, v6);
  v8 = *(__int64 **)(a2 + 64);
  v16 = *v7;
  v9 = &v8[*(unsigned int *)(a2 + 72)];
  if ( v8 == v9 )
    return 0;
  v17 = 0;
  v10 = *(__int64 **)(a2 + 64);
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v12 = *v10;
      v13 = sub_2EE8740(a1, *v10);
      if ( v13 )
      {
        v14 = *(_DWORD *)(v13 + 24) + v16;
        if ( !v11 || v14 < v17 )
          break;
      }
      if ( v9 == ++v10 )
        return v11;
    }
    ++v10;
    v17 = v14;
    v11 = v12;
  }
  while ( v9 != v10 );
  return v11;
}
