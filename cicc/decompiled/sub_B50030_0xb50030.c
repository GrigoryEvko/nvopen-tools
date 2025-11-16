// Function: sub_B50030
// Address: 0xb50030
//
__int64 __fastcall sub_B50030(__int64 a1, const void *a2, __int64 a3, __int64 a4)
{
  size_t v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax

  v5 = 4 * a3;
  v6 = (4 * a3) >> 2;
  v7 = *(unsigned int *)(a1 + 80);
  if ( v6 + v7 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, a1 + 88, v6 + v7, 4);
    v7 = *(unsigned int *)(a1 + 80);
  }
  if ( v5 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 72) + 4 * v7), a2, v5);
    LODWORD(v7) = *(_DWORD *)(a1 + 80);
  }
  *(_DWORD *)(a1 + 80) = v6 + v7;
  return sub_BD6B50(a1, a4);
}
