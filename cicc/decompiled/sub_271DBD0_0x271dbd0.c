// Function: sub_271DBD0
// Address: 0x271dbd0
//
__int64 __fastcall sub_271DBD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v4; // r14
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 result; // rax
  unsigned int v8; // eax
  __int64 v9; // rdx
  unsigned int v10; // eax

  sub_271D2D0((_BYTE *)a1);
  v4 = *(_BYTE *)(a1 + 2);
  if ( *(_BYTE *)(a2 + 12) )
  {
    v5 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v10 = sub_B6ED60(**(__int64 ***)a2, "clang.imprecise_release", 0x17u);
    *(_BYTE *)(a2 + 12) = 1;
    *(_DWORD *)(a2 + 8) = v10;
    v5 = v10;
  }
  if ( (_DWORD)v5 )
  {
    v6 = 0;
    if ( (*(_BYTE *)(a3 + 7) & 0x20) != 0 )
      v6 = sub_B91C10(a3, v5);
  }
  else
  {
    v6 = *(_QWORD *)(a3 + 48);
  }
  if ( v4 > 2u )
  {
    if ( v4 != 3 )
      BUG();
    goto LABEL_9;
  }
  result = 0;
  if ( v4 )
  {
    if ( v4 != 1 && !v6 )
      goto LABEL_9;
    ++*(_QWORD *)(a1 + 72);
    if ( !*(_BYTE *)(a1 + 100) )
    {
      v8 = 4 * (*(_DWORD *)(a1 + 92) - *(_DWORD *)(a1 + 96));
      v9 = *(unsigned int *)(a1 + 88);
      if ( v8 < 0x20 )
        v8 = 32;
      if ( (unsigned int)v9 > v8 )
      {
        sub_C8C990(a1 + 72, v5);
        goto LABEL_9;
      }
      memset(*(void **)(a1 + 80), -1, 8 * v9);
    }
    *(_QWORD *)(a1 + 92) = 0;
LABEL_9:
    *(_QWORD *)(a1 + 16) = v6;
    *(_BYTE *)(a1 + 9) = (*(_WORD *)(a3 + 2) & 3u) - 1 <= 1;
    return 1;
  }
  return result;
}
