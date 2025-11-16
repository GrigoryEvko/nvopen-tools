// Function: sub_2D55110
// Address: 0x2d55110
//
__int64 __fastcall sub_2D55110(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // rbx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r14
  __int64 v9; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  for ( i = *(_QWORD *)(a2 + 80); a2 + 72 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v7 = *(_QWORD *)(i + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == i + 24 )
      goto LABEL_16;
    if ( !v7 )
      BUG();
    v8 = v7 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_16:
      BUG();
    if ( *(_BYTE *)(v7 - 24) == 40 && *(_BYTE *)(*(_QWORD *)(v7 - 16) + 8LL) != 7 && *(_QWORD *)(v7 - 8) )
    {
      v9 = *(unsigned int *)(a1 + 8);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9 + 1, 8u, a5, a6);
        v9 = *(unsigned int *)(a1 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v9) = v8;
      ++*(_DWORD *)(a1 + 8);
    }
  }
  return a1;
}
