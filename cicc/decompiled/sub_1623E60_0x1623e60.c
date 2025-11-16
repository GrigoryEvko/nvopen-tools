// Function: sub_1623E60
// Address: 0x1623e60
//
unsigned __int64 __fastcall sub_1623E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // r13d
  unsigned int i; // ebx
  unsigned int v8; // esi
  unsigned __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // r13
  unsigned __int64 v12; // r13

  v6 = *(_DWORD *)(a1 + 8);
  if ( v6 )
  {
    for ( i = 0; i != v6; ++i )
    {
      v8 = i;
      result = sub_1623D00(a1, v8, 0);
    }
  }
  v10 = *(_QWORD *)(a1 + 16);
  if ( (v10 & 4) != 0 )
  {
    sub_161EF50((const __m128i *)(v10 & 0xFFFFFFFFFFFFFFF8LL), 0, a3, a4, a5);
    v11 = *(_QWORD *)(a1 + 16);
    if ( (v11 & 4) == 0 )
      BUG();
    v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)v12;
    if ( (*(_BYTE *)(v12 + 24) & 1) == 0 )
      j___libc_free_0(*(_QWORD *)(v12 + 32));
    return j_j___libc_free_0(v12, 128);
  }
  return result;
}
