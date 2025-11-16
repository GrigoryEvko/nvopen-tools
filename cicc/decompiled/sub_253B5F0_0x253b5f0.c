// Function: sub_253B5F0
// Address: 0x253b5f0
//
__int64 __fastcall sub_253B5F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rax

  if ( *(_BYTE *)a2 != 31 )
    return 1;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    return 1;
  v7 = *a1;
  v8 = *(unsigned int *)(*a1 + 8);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(*a1, (const void *)(v7 + 16), v8 + 1, 8u, a5, a6);
    v8 = *(unsigned int *)(v7 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v7 + 8 * v8) = a2;
  ++*(_DWORD *)(v7 + 8);
  return 1;
}
