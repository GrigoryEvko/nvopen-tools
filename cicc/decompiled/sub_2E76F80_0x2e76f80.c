// Function: sub_2E76F80
// Address: 0x2e76f80
//
__int64 __fastcall sub_2E76F80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r9
  void *v7; // rdi
  __int64 v8; // r8
  __int64 v9; // rbx
  unsigned int v10; // r14d
  unsigned __int16 *v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned int *v14; // r8
  unsigned int *i; // r10
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  int v19; // [rsp+4h] [rbp-3Ch]
  int v20; // [rsp+8h] [rbp-38h]

  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 16) + 200LL))(*(_QWORD *)(a3 + 16));
  v7 = (void *)(a1 + 16);
  v8 = *(unsigned int *)(v5 + 16);
  v9 = v5;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  v10 = (unsigned int)(v8 + 63) >> 6;
  if ( v10 > 6 )
  {
    v19 = v8;
    sub_C8D5F0(a1, (const void *)(a1 + 16), v10, 8u, v8, v6);
    memset(*(void **)a1, 0, 8LL * v10);
    *(_DWORD *)(a1 + 8) = v10;
    LODWORD(v8) = v19;
  }
  else
  {
    if ( v10 )
    {
      v20 = v8;
      memset(v7, 0, (size_t)v7 + 8 * v10 - a1 - 16);
      LODWORD(v8) = v20;
    }
    *(_DWORD *)(a1 + 8) = v10;
  }
  *(_DWORD *)(a1 + 64) = v8;
  if ( *(_BYTE *)(a2 + 120) )
  {
    v12 = (unsigned __int16 *)sub_2EBFBC0(*(_QWORD *)(a3 + 32));
    if ( v12 )
    {
      while ( 1 )
      {
        v13 = *v12;
        if ( !(_WORD)v13 )
          break;
        ++v12;
        *(_QWORD *)(*(_QWORD *)a1 + ((v13 >> 3) & 0x1FF8)) |= 1LL << v13;
      }
    }
    v14 = *(unsigned int **)(a2 + 96);
    for ( i = *(unsigned int **)(a2 + 104); i != v14; v14 += 3 )
    {
      v16 = *v14;
      v17 = v16;
      v18 = *(_QWORD *)(v9 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v9 + 8) + 24 * v16 + 4);
      if ( v18 )
      {
        while ( 1 )
        {
          v18 += 2;
          *(_QWORD *)(*(_QWORD *)a1 + ((v17 >> 3) & 0x1FF8)) &= ~(1LL << v17);
          if ( !*(_WORD *)(v18 - 2) )
            break;
          LODWORD(v16) = *(__int16 *)(v18 - 2) + (_DWORD)v16;
          v17 = (unsigned int)v16;
        }
      }
    }
  }
  return a1;
}
