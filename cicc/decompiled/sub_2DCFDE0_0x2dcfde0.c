// Function: sub_2DCFDE0
// Address: 0x2dcfde0
//
__int64 __fastcall sub_2DCFDE0(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  const void *v5; // r12
  int v6; // eax
  int v7; // eax
  size_t v9; // [rsp+0h] [rbp-40h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    v3 = i - 56;
    if ( !i )
      v3 = 0;
    if ( !sub_B2FC80(v3) && (*(_BYTE *)(v3 + 3) & 0x40) != 0 )
    {
      v4 = sub_B2DBE0(v3);
      v5 = *(const void **)v4;
      v9 = *(_QWORD *)(v4 + 8);
      v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      v6 = sub_C92610();
      v7 = sub_C92860((__int64 *)a1, v5, v9, v6);
      if ( v7 == -1 )
      {
        if ( *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) == v10 )
          return 1;
      }
      else if ( *(_QWORD *)a1 + 8LL * v7 == v10 )
      {
        return 1;
      }
    }
  }
  return 0;
}
