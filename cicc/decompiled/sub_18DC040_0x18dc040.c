// Function: sub_18DC040
// Address: 0x18dc040
//
__int64 __fastcall sub_18DC040(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v6; // r14
  int v7; // esi
  __int64 v8; // r12
  __int64 result; // rax
  int v10; // eax
  bool v11; // zf
  void *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rdx

  sub_18DB880((_BYTE *)a1);
  v6 = *(_BYTE *)(a1 + 2);
  if ( *(_BYTE *)(a2 + 12) )
  {
    v7 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v10 = sub_1602B80(**(__int64 ***)a2, "clang.imprecise_release", 0x17u);
    v11 = *(_BYTE *)(a2 + 12) == 0;
    *(_DWORD *)(a2 + 8) = v10;
    v7 = v10;
    if ( v11 )
    {
      *(_BYTE *)(a2 + 12) = 1;
      v8 = *(_QWORD *)(a3 + 48);
      if ( v8 )
        goto LABEL_4;
LABEL_12:
      if ( *(__int16 *)(a3 + 18) >= 0 )
        goto LABEL_5;
      goto LABEL_4;
    }
  }
  v8 = *(_QWORD *)(a3 + 48);
  if ( !v8 )
    goto LABEL_12;
LABEL_4:
  v8 = sub_1625790(a3, v7);
LABEL_5:
  if ( v6 <= 2u )
  {
    result = 0;
    if ( !v6 )
      return result;
    if ( v6 == 1 || v8 )
    {
      ++*(_QWORD *)(a1 + 80);
      v12 = *(void **)(a1 + 96);
      if ( v12 != *(void **)(a1 + 88) )
      {
        v13 = 4 * (*(_DWORD *)(a1 + 108) - *(_DWORD *)(a1 + 112));
        v14 = *(unsigned int *)(a1 + 104);
        if ( v13 < 0x20 )
          v13 = 32;
        if ( v13 < (unsigned int)v14 )
        {
          sub_16CC920(a1 + 80);
          goto LABEL_9;
        }
        memset(v12, -1, 8 * v14);
      }
      *(_QWORD *)(a1 + 108) = 0;
    }
  }
LABEL_9:
  *(_QWORD *)(a1 + 16) = v8;
  *(_BYTE *)(a1 + 9) = (*(_WORD *)(a3 + 18) & 3u) - 1 <= 1;
  return 1;
}
