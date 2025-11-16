// Function: sub_E5CB20
// Address: 0xe5cb20
//
__int64 __fastcall sub_E5CB20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  char v7; // dl
  __int64 v9; // rax

  v7 = *(_BYTE *)(a2 + 8) & 0x10;
  LOBYTE(v6) = v7 == 0;
  if ( v7 )
    return v6;
  *(_BYTE *)(a2 + 8) |= 0x10u;
  v9 = *(unsigned int *)(a1 + 64);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
  {
    sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v9 + 1, 8u, a5, a6);
    v9 = *(unsigned int *)(a1 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8 * v9) = a2;
  ++*(_DWORD *)(a1 + 64);
  return v6;
}
