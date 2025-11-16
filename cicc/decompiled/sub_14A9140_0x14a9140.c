// Function: sub_14A9140
// Address: 0x14a9140
//
__int64 __fastcall sub_14A9140(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdi
  __int64 result; // rax
  unsigned __int64 v7; // rdx
  int v8; // r14d
  size_t v9; // r9

  v5 = (_QWORD *)(a1 + 56);
  *(v5 - 7) = *(_QWORD *)a2;
  *(v5 - 6) = *(_QWORD *)(a2 + 8);
  *(v5 - 5) = *(_QWORD *)(a2 + 16);
  *(v5 - 4) = *(_QWORD *)(a2 + 24);
  *(v5 - 3) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = v5;
  *(_QWORD *)(a1 + 48) = 0x600000000LL;
  result = *(unsigned int *)(a2 + 104);
  *(_DWORD *)(a1 + 104) = result;
  if ( a1 + 40 != a2 + 40 )
  {
    v7 = *(unsigned int *)(a2 + 48);
    v8 = *(_DWORD *)(a2 + 48);
    if ( v8 )
    {
      v9 = 8 * v7;
      if ( v7 <= 6
        || (sub_16CD150(a1 + 40, v5, v7, 8), v5 = *(_QWORD **)(a1 + 40), (v9 = 8LL * *(unsigned int *)(a2 + 48)) != 0) )
      {
        memcpy(v5, *(const void **)(a2 + 40), v9);
        v5 = *(_QWORD **)(a1 + 40);
      }
      *(_DWORD *)(a1 + 48) = v8;
      result = *(unsigned int *)(a1 + 104);
    }
  }
  *(_DWORD *)(a1 + 104) = result + 1;
  v5[result] = a3;
  return result;
}
