// Function: sub_16E3CC0
// Address: 0x16e3cc0
//
__int64 __fastcall sub_16E3CC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
    v6 = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v6) = 0;
  ++*(_DWORD *)(a1 + 40);
  *(_BYTE *)(a1 + 95) = 1;
  return 0;
}
