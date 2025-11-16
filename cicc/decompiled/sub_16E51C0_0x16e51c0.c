// Function: sub_16E51C0
// Address: 0x16e51c0
//
void *__fastcall sub_16E51C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
    v6 = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v6) = 4;
  ++*(_DWORD *)(a1 + 40);
  sub_16E4E00(a1);
  *(_DWORD *)(a1 + 88) = *(_DWORD *)(a1 + 80);
  return sub_16E4B40(a1, "{ ", 2u);
}
