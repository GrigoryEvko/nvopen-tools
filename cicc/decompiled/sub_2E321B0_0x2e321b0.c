// Function: sub_2E321B0
// Address: 0x2e321b0
//
__int64 __fastcall sub_2E321B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx

  v6 = *(_QWORD *)(a1 + 144);
  if ( v6 != *(_QWORD *)(a1 + 152) )
    *(_QWORD *)(a1 + 152) = v6;
  v7 = *(unsigned int *)(a1 + 120);
  v8 = *(unsigned int *)(a1 + 124);
  if ( v7 + 1 > v8 )
  {
    sub_C8D5F0(a1 + 112, (const void *)(a1 + 128), v7 + 1, 8u, a5, a6);
    v7 = *(unsigned int *)(a1 + 120);
  }
  v9 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(v9 + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 120);
  return sub_2E32160(a2, a1, v9, v8, a5, a6);
}
