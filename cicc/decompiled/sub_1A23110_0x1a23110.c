// Function: sub_1A23110
// Address: 0x1a23110
//
__int64 __fastcall sub_1A23110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax
  __int64 result; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned int v11; // ebx
  unsigned __int64 v12; // r15
  unsigned int v13; // r14d
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  int v18; // eax
  unsigned int v19; // r14d
  __int64 v20; // rbx

  v6 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v6 + 16) )
    BUG();
  result = *(unsigned int *)(v6 + 36);
  if ( (_DWORD)result == 4 )
  {
    v20 = *(_QWORD *)(a1 + 376);
    result = *(unsigned int *)(v20 + 304);
    if ( (unsigned int)result >= *(_DWORD *)(v20 + 308) )
    {
      sub_16CD150(v20 + 296, (const void *)(v20 + 312), 0, 8, a5, a6);
      result = *(unsigned int *)(v20 + 304);
    }
    *(_QWORD *)(*(_QWORD *)(v20 + 296) + 8 * result) = *(_QWORD *)(a1 + 336);
    ++*(_DWORD *)(v20 + 304);
  }
  else if ( *(_BYTE *)(a1 + 344) )
  {
    if ( (unsigned int)(result - 116) <= 1 )
    {
      v10 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v11 = *(_DWORD *)(v10 + 32);
      if ( v11 <= 0x40 )
      {
        v12 = *(_QWORD *)(v10 + 24);
      }
      else
      {
        v12 = -1;
        if ( v11 - (unsigned int)sub_16A57B0(v10 + 24) <= 0x40 )
          v12 = **(_QWORD **)(v10 + 24);
      }
      v13 = *(_DWORD *)(a1 + 360);
      v14 = *(_QWORD *)(a1 + 368);
      v15 = a1 + 352;
      if ( v13 > 0x40 )
      {
        v18 = sub_16A57B0(a1 + 352);
        v15 = a1 + 352;
        v19 = v13 - v18;
        v16 = -1;
        if ( v19 <= 0x40 )
          v16 = **(_QWORD **)(a1 + 352);
      }
      else
      {
        v16 = *(_QWORD *)(a1 + 352);
      }
      v17 = v14 - v16;
      if ( v14 - v16 > v12 )
        v17 = v12;
      return sub_1A22CF0((_QWORD *)a1, a2, v15, v17, 1u, a6);
    }
    else
    {
      v8 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = v8 | *(_QWORD *)(a1 + 8) & 3LL | 4;
      result = v8 | *(_QWORD *)(a1 + 16) & 3LL | 4;
      *(_QWORD *)(a1 + 16) = result;
      *(_QWORD *)(a1 + 8) = v9;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a2 | *(_QWORD *)(a1 + 8) & 3LL | 4;
  }
  return result;
}
