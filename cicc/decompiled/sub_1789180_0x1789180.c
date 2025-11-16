// Function: sub_1789180
// Address: 0x1789180
//
__int64 __fastcall sub_1789180(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r12
  __int64 *v6; // rax
  char v7; // dl
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v12; // rdx
  __int64 *v13; // rsi
  __int64 *v14; // rcx

  v5 = a1;
  v6 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v6 )
    goto LABEL_2;
  v12 = *(unsigned int *)(a3 + 28);
  v13 = &v6[v12];
  if ( v6 != v13 )
  {
    v14 = 0;
    while ( a1 != *v6 )
    {
      if ( *v6 == -2 )
        v14 = v6;
      if ( v13 == ++v6 )
      {
        if ( !v14 )
          goto LABEL_23;
        *v14 = a1;
        --*(_DWORD *)(a3 + 32);
        ++*(_QWORD *)a3;
        goto LABEL_3;
      }
    }
    return 1;
  }
LABEL_23:
  if ( (unsigned int)v12 < *(_DWORD *)(a3 + 24) )
  {
    *(_DWORD *)(a3 + 28) = v12 + 1;
    *v13 = a1;
    ++*(_QWORD *)a3;
  }
  else
  {
LABEL_2:
    sub_16CCBA0(a3, a1);
    if ( !v7 )
      return 1;
  }
LABEL_3:
  if ( *(_DWORD *)(a3 + 28) - *(_DWORD *)(a3 + 32) == 16 )
    return 0;
  v8 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v9 = *(_QWORD *)(a1 - 8);
    v5 = v9 + v8;
  }
  else
  {
    v9 = a1 - v8;
  }
  for ( ; v5 != v9; v9 += 24 )
  {
    v10 = *(_QWORD *)v9;
    if ( *(_BYTE *)(*(_QWORD *)v9 + 16LL) == 77 )
    {
      if ( !(unsigned __int8)sub_1789180(v10, a2, a3) )
        return 0;
    }
    else if ( a2 != v10 )
    {
      return 0;
    }
  }
  return 1;
}
