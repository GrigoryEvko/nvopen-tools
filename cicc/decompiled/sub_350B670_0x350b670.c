// Function: sub_350B670
// Address: 0x350b670
//
unsigned __int64 __fastcall sub_350B670(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        char a7,
        unsigned int a8,
        __int64 a9)
{
  int v10; // r13d
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v22; // rax

  v10 = a4;
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, __int64, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 48) + 240LL))(
    *(_QWORD *)(a1 + 48),
    a2,
    a3,
    a4,
    a8,
    a5[1],
    a6);
  v12 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v12 )
    BUG();
  v13 = *(_QWORD *)v12;
  if ( (*(_QWORD *)v12 & 4) == 0 && (*(_BYTE *)(v12 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v22 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = v22;
      if ( (*(_BYTE *)(v22 + 44) & 4) == 0 )
        break;
      v13 = *(_QWORD *)v22;
    }
  }
  sub_2E8D7A0(v12, v10);
  v18 = *a5;
  if ( !*(_BYTE *)(a1 + 172) )
  {
LABEL_11:
    sub_C8CC70(a1 + 144, v18, (__int64)v14, v15, v16, v17);
    v20 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    if ( !a9 )
      return sub_2E192D0(v20, v12, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
    return sub_2F6A220(v20, a9, v12) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
  v19 = *(__int64 **)(a1 + 152);
  v15 = *(unsigned int *)(a1 + 164);
  v14 = &v19[v15];
  if ( v19 == v14 )
  {
LABEL_10:
    if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 160) )
      goto LABEL_11;
    *(_DWORD *)(a1 + 164) = v15 + 1;
    *v14 = v18;
    ++*(_QWORD *)(a1 + 144);
  }
  else
  {
    while ( v18 != *v19 )
    {
      if ( v14 == ++v19 )
        goto LABEL_10;
    }
  }
  v20 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
  if ( a9 )
    return sub_2F6A220(v20, a9, v12) & 0xFFFFFFFFFFFFFFF8LL | 4;
  return sub_2E192D0(v20, v12, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
}
