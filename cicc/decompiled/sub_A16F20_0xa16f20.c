// Function: sub_A16F20
// Address: 0xa16f20
//
__int64 __fastcall sub_A16F20(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // r13
  unsigned int v4; // ebx
  int v5; // ebx
  __int64 *v6; // r14
  __int64 v7; // rbx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r8

  v2 = 1;
  v3 = (__int64 *)a2;
  v4 = *(_DWORD *)(a2 + 8);
  if ( v4 > 0x40 )
  {
    v5 = v4 - sub_C444A0(a2);
    v2 = 1;
    if ( v5 )
      v2 = ((unsigned int)(v5 - 1) >> 6) + 1;
    v3 = *(__int64 **)a2;
  }
  v6 = &v3[v2];
  do
  {
    v9 = *((unsigned int *)a1 + 2);
    v10 = *v3;
    v11 = *((unsigned int *)a1 + 3);
    v12 = v9 + 1;
    if ( *v3 >= 0 )
    {
      v7 = 2 * v10;
      if ( v12 <= v11 )
        goto LABEL_7;
    }
    else
    {
      v7 = -2 * v10 + 1;
      if ( v12 <= v11 )
        goto LABEL_7;
    }
    sub_C8D5F0(a1, a1 + 2, v9 + 1, 8);
    v9 = *((unsigned int *)a1 + 2);
LABEL_7:
    result = *a1;
    ++v3;
    *(_QWORD *)(*a1 + 8 * v9) = v7;
    ++*((_DWORD *)a1 + 2);
  }
  while ( v6 != v3 );
  return result;
}
