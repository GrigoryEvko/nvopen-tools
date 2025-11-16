// Function: sub_20F81F0
// Address: 0x20f81f0
//
unsigned int *__fastcall sub_20F81F0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rcx
  __int64 v3; // r9
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned int *v6; // r14

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 24) + a2) <= 0x1Fu
    && a2 == *(_DWORD *)(a1 + 720LL * *(unsigned __int8 *)(*(_QWORD *)(a1 + 24) + a2) + 48) )
  {
    v6 = (unsigned int *)(a1 + 720LL * *(unsigned __int8 *)(*(_QWORD *)(a1 + 24) + a2) + 48);
    if ( !sub_20F7B50(v6, *(_QWORD *)(a1 + 8), *(_QWORD *)a1) )
      sub_20F7AB0(v6, *(_QWORD *)(a1 + 8), *(_QWORD *)a1);
  }
  else
  {
    v4 = *(_DWORD *)(a1 + 40);
    if ( v4 == 31 )
    {
      *(_DWORD *)(a1 + 40) = 0;
      v4 = 31;
    }
    else
    {
      *(_DWORD *)(a1 + 40) = v4 + 1;
    }
    while ( 1 )
    {
      v5 = 720LL * v4;
      if ( !*(_DWORD *)(a1 + v5 + 56) )
        break;
      if ( ++v4 == 32 )
        v4 = 0;
    }
    v6 = (unsigned int *)(a1 + v5 + 48);
    sub_20F7DC0(v6, a2, v3, v2, *(_QWORD *)(a1 + 16), v3);
    *(_BYTE *)(*(_QWORD *)(a1 + 24) + a2) = v4;
  }
  return v6;
}
