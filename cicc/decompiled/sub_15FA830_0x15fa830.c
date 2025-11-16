// Function: sub_15FA830
// Address: 0x15fa830
//
__int64 __fastcall sub_15FA830(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v5; // r14
  unsigned __int8 v7; // al
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rdx
  unsigned __int64 v11; // r15
  unsigned int v12; // r12d
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  char v15; // al
  int v16; // r15d
  unsigned __int64 v17; // r14
  unsigned int v18; // ebx
  int v19; // eax
  __int64 v20; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    return 0;
  if ( *a2 != *(_QWORD *)a1 )
    return 0;
  v5 = *(_QWORD *)a3;
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 16 )
    return 0;
  v3 = sub_1642F90(*(_QWORD *)(v5 + 24), 32);
  if ( !(_BYTE)v3 )
    return 0;
  v7 = *(_BYTE *)(a3 + 16);
  if ( (unsigned __int8)(v7 - 9) <= 1u )
    return v3;
  if ( v7 != 8 )
  {
    if ( (unsigned int)v7 - 11 > 1 )
    {
      if ( v7 == 5 )
      {
        LOBYTE(v3) = *(_WORD *)(a3 + 18) == 56;
        return v3;
      }
    }
    else
    {
      v16 = *(_QWORD *)(v5 + 32);
      if ( !v16 )
        return v3;
      v17 = 2 * (unsigned int)*(_QWORD *)(*(_QWORD *)a1 + 32LL);
      v18 = 0;
      while ( sub_1595A50(a3, v18) < v17 )
      {
        if ( v16 == ++v18 )
          return v3;
      }
    }
    return 0;
  }
  v8 = 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v9 = *(_QWORD *)(a3 - 8);
    v10 = v9 + v8;
  }
  else
  {
    v10 = a3;
    v9 = a3 - v8;
  }
  if ( v9 != v10 )
  {
    v11 = 2 * (unsigned int)*(_QWORD *)(*(_QWORD *)a1 + 32LL);
    do
    {
      v14 = *(_QWORD *)v9;
      v15 = *(_BYTE *)(*(_QWORD *)v9 + 16LL);
      if ( v15 == 13 )
      {
        v12 = *(_DWORD *)(v14 + 32);
        if ( v12 > 0x40 )
        {
          v20 = v10;
          v19 = sub_16A57B0(v14 + 24);
          v10 = v20;
          if ( v12 - v19 > 0x40 )
            return 0;
          v13 = **(_QWORD **)(v14 + 24);
        }
        else
        {
          v13 = *(_QWORD *)(v14 + 24);
        }
        if ( v11 <= v13 )
          return 0;
      }
      else if ( v15 != 9 )
      {
        return 0;
      }
      v9 += 24;
    }
    while ( v10 != v9 );
  }
  return v3;
}
