// Function: sub_18DBB50
// Address: 0x18dbb50
//
__int64 __fastcall sub_18DBB50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  int v6; // r8d
  __int64 v7; // rbx
  char v8; // si
  char v9; // al
  __int64 *v10; // rax
  __int64 *v12; // rsi
  unsigned int v13; // edi
  __int64 *v14; // rcx
  int v15; // eax
  bool v16; // zf

  LOBYTE(v3) = (unsigned int)*(unsigned __int8 *)(a1 + 2) - 5 <= 1;
  if ( *(_BYTE *)(a2 + 12) )
  {
    v6 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v15 = sub_1602B80(**(__int64 ***)a2, "clang.imprecise_release", 0x17u);
    v16 = *(_BYTE *)(a2 + 12) == 0;
    *(_DWORD *)(a2 + 8) = v15;
    v6 = v15;
    if ( v16 )
    {
      *(_BYTE *)(a2 + 12) = 1;
      v7 = *(_QWORD *)(a3 + 48);
      if ( v7 )
        goto LABEL_4;
      goto LABEL_18;
    }
  }
  v7 = *(_QWORD *)(a3 + 48);
  if ( v7 )
  {
LABEL_4:
    v7 = sub_1625790(a3, v6);
    v8 = 5 - ((v7 == 0) - 1);
    goto LABEL_5;
  }
LABEL_18:
  v8 = 5;
  if ( *(__int16 *)(a3 + 18) < 0 )
    goto LABEL_4;
LABEL_5:
  sub_18DB9D0(a1, v8);
  v9 = *(_BYTE *)a1;
  *(_QWORD *)(a1 + 16) = v7;
  *(_BYTE *)(a1 + 8) = v9;
  v10 = *(__int64 **)(a1 + 32);
  *(_BYTE *)(a1 + 9) = (*(_WORD *)(a3 + 18) & 3u) - 1 <= 1;
  if ( *(__int64 **)(a1 + 40) != v10 )
  {
LABEL_6:
    sub_16CCBA0(a1 + 24, a3);
    goto LABEL_7;
  }
  v12 = &v10[*(unsigned int *)(a1 + 52)];
  v13 = *(_DWORD *)(a1 + 52);
  if ( v10 == v12 )
  {
LABEL_20:
    if ( v13 < *(_DWORD *)(a1 + 48) )
    {
      *(_DWORD *)(a1 + 52) = v13 + 1;
      *v12 = a3;
      ++*(_QWORD *)(a1 + 24);
      goto LABEL_7;
    }
    goto LABEL_6;
  }
  v14 = 0;
  while ( a3 != *v10 )
  {
    if ( *v10 == -2 )
      v14 = v10;
    if ( v12 == ++v10 )
    {
      if ( !v14 )
        goto LABEL_20;
      *v14 = a3;
      --*(_DWORD *)(a1 + 56);
      ++*(_QWORD *)(a1 + 24);
      break;
    }
  }
LABEL_7:
  sub_18DB870((_BYTE *)a1);
  return v3;
}
