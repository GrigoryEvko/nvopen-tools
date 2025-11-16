// Function: sub_1C13660
// Address: 0x1c13660
//
void __fastcall sub_1C13660(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  int v6; // r9d
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  void *v9; // rdi
  size_t v10; // rdx
  unsigned __int64 *v11; // rdx

  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 16;
  *(_QWORD *)a1 = &unk_49F7548;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 1;
  sub_1C12880(0);
  *(_QWORD *)(a1 + 32) = a1 + 16;
  v7 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 16;
  *(_QWORD *)(a1 + 48) = 0;
  if ( v7 == *(_QWORD *)(a1 + 64) )
  {
    v8 = *(unsigned int *)(a2 + 8);
    if ( !*(_DWORD *)(a2 + 8) )
      goto LABEL_5;
  }
  else
  {
    v8 = *(unsigned int *)(a2 + 8);
    *(_QWORD *)(a1 + 64) = v7;
    if ( !v8 )
      goto LABEL_5;
  }
  sub_14F2040(a1 + 56, v8);
  v9 = *(void **)(a1 + 56);
  v10 = 8LL * *(unsigned int *)(a2 + 8);
  if ( v10 )
    memmove(v9, *(const void **)a2, v10);
LABEL_5:
  v11 = *(unsigned __int64 **)(a2 + 80);
  if ( v11 )
    sub_1C12D00((_QWORD *)a1, a2, v11, v4, v5, v6);
}
