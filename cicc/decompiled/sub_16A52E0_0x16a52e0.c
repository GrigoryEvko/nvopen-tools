// Function: sub_16A52E0
// Address: 0x16a52e0
//
void __fastcall sub_16A52E0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // ebx
  unsigned int v5; // r10d
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned int v10; // r8d
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned int v13; // ecx
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  const void *v19; // rsi
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx

  v4 = *(_DWORD *)(a2 + 8);
  v5 = *(_DWORD *)(a1 + 8);
  if ( v5 == v4 )
  {
    if ( v5 > 0x40 )
    {
      sub_16A51C0(a1, a2);
    }
    else
    {
      v22 = *(_QWORD *)a2;
      *(_QWORD *)a1 = v22;
      v23 = *(unsigned int *)(a2 + 8);
      *(_DWORD *)(a1 + 8) = v23;
      v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v23;
      if ( (unsigned int)v23 > 0x40 )
      {
        v25 = (unsigned int)((unsigned __int64)(v23 + 63) >> 6) - 1;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v25) &= v24;
      }
      else
      {
        *(_QWORD *)a1 = v22 & v24;
      }
    }
  }
  else if ( v5 <= 0x40 )
  {
    v17 = *(_QWORD *)a1 & ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << a3);
    *(_QWORD *)a1 = v17;
    *(_QWORD *)a1 = (*(_QWORD *)a2 << a3) | v17;
  }
  else
  {
    v7 = a3 & 0x3F;
    v8 = a3 >> 6;
    v9 = (v4 + a3 - 1) >> 6;
    if ( (_DWORD)v8 == (_DWORD)v9 )
    {
      v18 = 8 * v8;
      *(_QWORD *)(v18 + *(_QWORD *)a1) &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << v7);
      *(_QWORD *)(*(_QWORD *)a1 + v18) |= *(_QWORD *)a2 << v7;
    }
    else if ( v7 )
    {
      if ( v4 )
      {
        v10 = v4;
        v11 = 0;
        while ( 1 )
        {
          v12 = *(_QWORD *)a2;
          if ( v10 > 0x40 )
            v12 = *(_QWORD *)(v12 + 8LL * (v11 >> 6));
          v13 = a3 + v11;
          v14 = *(_QWORD *)a1;
          v15 = 1LL << ((unsigned __int8)a3 + (unsigned __int8)v11);
          if ( (v12 & (1LL << v11)) != 0 )
            break;
          v16 = ~v15;
          if ( v5 > 0x40 )
          {
            *(_QWORD *)(v14 + 8LL * (v13 >> 6)) &= v16;
            goto LABEL_9;
          }
          ++v11;
          *(_QWORD *)a1 = v14 & v16;
          if ( v4 == v11 )
            return;
LABEL_10:
          v10 = *(_DWORD *)(a2 + 8);
          v5 = *(_DWORD *)(a1 + 8);
        }
        if ( v5 > 0x40 )
          *(_QWORD *)(v14 + 8LL * (v13 >> 6)) |= v15;
        else
          *(_QWORD *)a1 = v14 | v15;
LABEL_9:
        if ( v4 == ++v11 )
          return;
        goto LABEL_10;
      }
    }
    else
    {
      v19 = (const void *)a2;
      if ( v4 > 0x40 )
        v19 = *(const void **)a2;
      memcpy((void *)(*(_QWORD *)a1 + 8 * v8), v19, 8 * (v4 >> 6));
      if ( (v4 & 0x3F) != 0 )
      {
        v20 = 8 * v9;
        *(_QWORD *)(v20 + *(_QWORD *)a1) &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (v4 & 0x3F)));
        if ( *(_DWORD *)(a2 + 8) <= 0x40u )
          v21 = *(_QWORD *)a2;
        else
          v21 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v4 - 1) >> 6));
        *(_QWORD *)(*(_QWORD *)a1 + v20) |= v21;
      }
    }
  }
}
