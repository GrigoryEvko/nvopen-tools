// Function: sub_C43D80
// Address: 0xc43d80
//
void __fastcall sub_C43D80(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // ebx
  unsigned int v5; // r10d
  __int64 v7; // rdi
  char v9; // r11
  __int64 v10; // rdx
  __int64 v11; // r14
  int v12; // r11d
  unsigned int v13; // r9d
  unsigned int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // ecx
  __int64 v17; // r9
  __int64 v18; // r9
  unsigned __int64 v19; // rdi
  const void *v20; // rsi
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx

  v4 = *(_DWORD *)(a2 + 8);
  if ( !v4 )
    return;
  v5 = *(_DWORD *)(a1 + 8);
  if ( v5 == v4 )
  {
    if ( v5 > 0x40 )
    {
      sub_C43990(a1, a2);
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    }
    return;
  }
  v7 = *(_QWORD *)a1;
  if ( v5 <= 0x40 )
  {
    v19 = ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << a3) & v7;
    *(_QWORD *)a1 = v19;
    *(_QWORD *)a1 = (*(_QWORD *)a2 << a3) | v19;
    return;
  }
  v9 = a3;
  v10 = a3 >> 6;
  v11 = (v4 + a3 - 1) >> 6;
  v12 = v9 & 0x3F;
  if ( (_DWORD)v10 == (_DWORD)v11 )
  {
    v23 = 8 * v10;
    *(_QWORD *)(v7 + v23) &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << v12);
    *(_QWORD *)(*(_QWORD *)a1 + v23) |= *(_QWORD *)a2 << v12;
    return;
  }
  v13 = v4;
  v14 = 0;
  if ( v12 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)a2;
      if ( v13 > 0x40 )
        v15 = *(_QWORD *)(v15 + 8LL * (v14 >> 6));
      v16 = a3 + v14;
      v17 = 1LL << ((unsigned __int8)a3 + (unsigned __int8)v14);
      if ( (v15 & (1LL << v14)) != 0 )
        break;
      v18 = ~v17;
      if ( v5 > 0x40 )
      {
        *(_QWORD *)(v7 + 8LL * (v16 >> 6)) &= v18;
        goto LABEL_9;
      }
      ++v14;
      *(_QWORD *)a1 = v7 & v18;
      if ( v4 == v14 )
        return;
LABEL_10:
      v13 = *(_DWORD *)(a2 + 8);
      v5 = *(_DWORD *)(a1 + 8);
      v7 = *(_QWORD *)a1;
    }
    if ( v5 > 0x40 )
      *(_QWORD *)(v7 + 8LL * (v16 >> 6)) |= v17;
    else
      *(_QWORD *)a1 = v17 | v7;
LABEL_9:
    if ( v4 == ++v14 )
      return;
    goto LABEL_10;
  }
  v20 = (const void *)a2;
  if ( v4 > 0x40 )
    v20 = *(const void **)a2;
  memcpy((void *)(v7 + 8 * v10), v20, 8 * (v4 >> 6));
  if ( (v4 & 0x3F) != 0 )
  {
    v21 = 8 * v11;
    *(_QWORD *)(v21 + *(_QWORD *)a1) &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (v4 & 0x3F)));
    if ( *(_DWORD *)(a2 + 8) <= 0x40u )
      v22 = *(_QWORD *)a2;
    else
      v22 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v4 - 1) >> 6));
    *(_QWORD *)(*(_QWORD *)a1 + v21) |= v22;
  }
}
