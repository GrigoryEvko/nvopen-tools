// Function: sub_25DDDB0
// Address: 0x25dddb0
//
__int64 __fastcall sub_25DDDB0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r8d
  _QWORD *v4; // rdi
  _QWORD *v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 *v9; // rax

  v3 = *(unsigned __int8 *)(a1 + 28);
  if ( (_BYTE)v3 )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = &v4[*(unsigned int *)(a1 + 20)];
    v6 = v4;
    if ( v4 == v5 )
      return 0;
    while ( *v6 != a2 )
    {
      if ( v5 == ++v6 )
        return 0;
    }
    v7 = (unsigned int)(*(_DWORD *)(a1 + 20) - 1);
    *(_DWORD *)(a1 + 20) = v7;
    *v6 = v4[v7];
    ++*(_QWORD *)a1;
    return v3;
  }
  else
  {
    v9 = sub_C8CA60(a1, a2);
    if ( !v9 )
      return 0;
    *v9 = -2;
    ++*(_DWORD *)(a1 + 24);
    ++*(_QWORD *)a1;
    return 1;
  }
}
