// Function: sub_CB2560
// Address: 0xcb2560
//
void *__fastcall sub_CB2560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a1 + 40);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v6 + 1, 4u, a5, a6);
    v6 = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v6) = 6;
  ++*(_DWORD *)(a1 + 40);
  sub_CB20A0(a1, 0);
  *(_DWORD *)(a1 + 88) = *(_DWORD *)(a1 + 80);
  return sub_CB1B10(a1, "{ ", 2u);
}
