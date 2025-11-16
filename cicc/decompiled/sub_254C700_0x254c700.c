// Function: sub_254C700
// Address: 0x254c700
//
__int64 __fastcall sub_254C700(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 result; // rax
  void *v6; // rax
  __int64 v7; // rdx
  const void *v8; // rsi

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  sub_C7D6A0(0, 0, 8);
  v4 = *(unsigned int *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v4;
  if ( (_DWORD)v4 )
  {
    v6 = (void *)sub_C7D670(8 * v4, 8);
    v7 = *(unsigned int *)(a1 + 24);
    v8 = *(const void **)(a2 + 8);
    *(_QWORD *)(a1 + 8) = v6;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    memcpy(v6, v8, 8 * v7);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
  }
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  result = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 56) = result;
  return result;
}
