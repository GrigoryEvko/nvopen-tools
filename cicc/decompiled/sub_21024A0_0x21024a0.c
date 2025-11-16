// Function: sub_21024A0
// Address: 0x21024a0
//
unsigned __int64 __fastcall sub_21024A0(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        char a7)
{
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // rax
  unsigned __int64 v15; // rax
  __int64 *v16; // rdi
  unsigned int v17; // r8d
  __int64 *v18; // rcx

  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, __int64, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 48) + 152LL))(
    *(_QWORD *)(a1 + 48),
    a2,
    a3,
    a4,
    0,
    a5[1],
    a6);
  v10 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v10 )
    BUG();
  v11 = *(_QWORD *)v10;
  if ( (*(_QWORD *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v15 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = v15;
      if ( (*(_BYTE *)(v15 + 46) & 4) == 0 )
        break;
      v11 = *(_QWORD *)v15;
    }
  }
  *(_BYTE *)(*(_QWORD *)(v10 + 32) + 3LL) &= ~0x40u;
  v12 = *a5;
  v13 = *(__int64 **)(a1 + 160);
  if ( *(__int64 **)(a1 + 168) != v13 )
  {
LABEL_4:
    sub_16CCBA0(a1 + 152, v12);
    return sub_1DC1550(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL), v10, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
  v16 = &v13[*(unsigned int *)(a1 + 180)];
  v17 = *(_DWORD *)(a1 + 180);
  if ( v13 == v16 )
  {
LABEL_19:
    if ( v17 < *(_DWORD *)(a1 + 176) )
    {
      *(_DWORD *)(a1 + 180) = v17 + 1;
      *v16 = v12;
      ++*(_QWORD *)(a1 + 152);
      return sub_1DC1550(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL), v10, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
    }
    goto LABEL_4;
  }
  v18 = 0;
  while ( v12 != *v13 )
  {
    if ( *v13 == -2 )
      v18 = v13;
    if ( v16 == ++v13 )
    {
      if ( !v18 )
        goto LABEL_19;
      *v18 = v12;
      --*(_DWORD *)(a1 + 184);
      ++*(_QWORD *)(a1 + 152);
      return sub_1DC1550(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL), v10, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
    }
  }
  return sub_1DC1550(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL), v10, a7) & 0xFFFFFFFFFFFFFFF8LL | 4;
}
