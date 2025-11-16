// Function: sub_B972A0
// Address: 0xb972a0
//
void __fastcall sub_B972A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // r13d
  unsigned int i; // ebx
  unsigned int v8; // esi
  __int64 v9; // rdi
  __int64 v10; // r13
  unsigned __int64 v11; // r13

  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v6 = *(_DWORD *)(a1 - 24);
  else
    v6 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  if ( v6 )
  {
    for ( i = 0; i != v6; ++i )
    {
      v8 = i;
      sub_B97110(a1, v8, 0);
    }
  }
  v9 = *(_QWORD *)(a1 + 8);
  if ( (v9 & 4) != 0 )
  {
    sub_B92F50((const __m128i *)(v9 & 0xFFFFFFFFFFFFFFF8LL), 0, a3, a4, a5);
    v10 = *(_QWORD *)(a1 + 8);
    if ( (v10 & 4) == 0 )
      BUG();
    v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 8) = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFFBLL;
    if ( (*(_BYTE *)(v11 + 24) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v11 + 32), 24LL * *(unsigned int *)(v11 + 40), 8);
    j_j___libc_free_0(v11, 128);
  }
}
