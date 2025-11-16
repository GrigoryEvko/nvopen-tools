// Function: sub_15FB110
// Address: 0x15fb110
//
__int64 __fastcall sub_15FB110(__int64 a1, const void *a2, __int64 a3, __int64 a4)
{
  size_t v5; // r13
  unsigned __int64 v6; // rbx
  __int64 v7; // rdx

  v5 = 4 * a3;
  v6 = (4 * a3) >> 2;
  v7 = *(unsigned int *)(a1 + 64);
  if ( v6 > (unsigned __int64)*(unsigned int *)(a1 + 68) - v7 )
  {
    sub_16CD150(a1 + 56, a1 + 72, v6 + v7, 4);
    v7 = *(unsigned int *)(a1 + 64);
  }
  if ( v5 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 56) + 4 * v7), a2, v5);
    LODWORD(v7) = *(_DWORD *)(a1 + 64);
  }
  *(_DWORD *)(a1 + 64) = v6 + v7;
  return sub_164B780(a1, a4);
}
