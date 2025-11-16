// Function: sub_1E80340
// Address: 0x1e80340
//
__int64 __fastcall sub_1E80340(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r8d
  int *v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // r15
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned int v12; // ecx
  int v14; // [rsp+8h] [rbp-38h]
  unsigned int v15; // [rsp+Ch] [rbp-34h]

  if ( *(_QWORD *)(a2 + 64) == *(_QWORD *)(a2 + 72) )
    return 0;
  v2 = sub_1E80290(a1, a2);
  if ( v2 )
  {
    if ( a2 == **(_QWORD **)(v2 + 32) )
      return 0;
  }
  v6 = (int *)sub_1E7FE90(*(_QWORD *)(a1 + 440), a2, v3, v4, v5);
  v7 = *(__int64 **)(a2 + 72);
  v14 = *v6;
  if ( v7 == *(__int64 **)(a2 + 64) )
    return 0;
  v15 = 0;
  v8 = *(__int64 **)(a2 + 64);
  v9 = 0;
  do
  {
    while ( 1 )
    {
      v10 = *v8;
      v11 = sub_1E80310(a1, *v8);
      if ( v11 )
      {
        v12 = *(_DWORD *)(v11 + 24) + v14;
        if ( !v9 || v12 < v15 )
          break;
      }
      if ( v7 == ++v8 )
        return v9;
    }
    ++v8;
    v15 = v12;
    v9 = v10;
  }
  while ( v7 != v8 );
  return v9;
}
