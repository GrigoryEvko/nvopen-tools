// Function: sub_2C4F040
// Address: 0x2c4f040
//
__int64 __fastcall sub_2C4F040(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rax
  _QWORD *v7; // rcx
  __int64 *v8; // rax
  unsigned int v9; // [rsp+Ch] [rbp-14h]

  if ( (unsigned int)**(unsigned __int8 **)(a2 - 32) - 12 > 1 )
    return *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3);
  v3 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v3 != 92 )
    return *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3);
  v5 = *a1;
  if ( *(_BYTE *)(v5 + 28) )
  {
    v6 = *(_QWORD **)(v5 + 8);
    v7 = &v6[*(unsigned int *)(v5 + 20)];
    if ( v6 == v7 )
      return *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3);
    while ( v3 != *v6 )
    {
      if ( v7 == ++v6 )
        return *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3);
    }
  }
  else
  {
    v9 = a3;
    v8 = sub_C8CA60(v5, *(_QWORD *)(a2 - 64));
    a3 = v9;
    if ( !v8 )
      return *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3);
  }
  return *(unsigned int *)(*(_QWORD *)(v3 + 72) + 4LL * *(unsigned int *)(*(_QWORD *)(a2 + 72) + 4LL * a3));
}
