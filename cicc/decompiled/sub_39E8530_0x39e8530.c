// Function: sub_39E8530
// Address: 0x39e8530
//
void __fastcall sub_39E8530(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rdi
  size_t v9; // r12

  v8 = *(unsigned int *)(a1 + 8);
  v9 = a3 - a2;
  if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v8 < a3 - a2 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), v8 + v9, 1, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  if ( a2 != a3 )
  {
    memcpy((void *)(*(_QWORD *)a1 + v8), a2, v9);
    LODWORD(v8) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v9 + v8;
}
