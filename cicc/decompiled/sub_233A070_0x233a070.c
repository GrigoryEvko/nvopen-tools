// Function: sub_233A070
// Address: 0x233a070
//
void *__fastcall sub_233A070(__int64 a1, __int64 a2)
{
  __int16 v4; // ax
  void *result; // rax
  __int64 v6; // rdi
  void *v7; // rax
  __int64 v8; // rdx
  const void *v9; // rsi

  *(_DWORD *)a1 = *(_DWORD *)a2;
  v4 = *(_WORD *)(a2 + 4);
  *(_QWORD *)(a1 + 8) = 0;
  *(_WORD *)(a1 + 4) = v4;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 0;
  result = (void *)sub_C7D6A0(0, 0, 4);
  v6 = *(unsigned int *)(a2 + 32);
  *(_DWORD *)(a1 + 32) = v6;
  if ( (_DWORD)v6 )
  {
    v7 = (void *)sub_C7D670(4 * v6, 4);
    v8 = *(unsigned int *)(a1 + 32);
    v9 = *(const void **)(a2 + 16);
    *(_QWORD *)(a1 + 16) = v7;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
    return memcpy(v7, v9, 4 * v8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
