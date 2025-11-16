// Function: sub_1E0CC60
// Address: 0x1e0cc60
//
__int64 __fastcall sub_1E0CC60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  __int64 v8; // rbx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 result; // rax

  v8 = sub_1E0C9D0(a1, a2, a3, a4, a5, a6);
  v11 = *(unsigned int *)(v8 + 16);
  if ( (unsigned int)v11 >= *(_DWORD *)(v8 + 20) )
  {
    sub_16CD150(v8 + 8, (const void *)(v8 + 24), 0, 8, v9, v10);
    v11 = *(unsigned int *)(v8 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8 * v11) = a3;
  result = *(unsigned int *)(v8 + 40);
  ++*(_DWORD *)(v8 + 16);
  if ( (unsigned int)result >= *(_DWORD *)(v8 + 44) )
  {
    sub_16CD150(v8 + 32, (const void *)(v8 + 48), 0, 8, v9, v10);
    result = *(unsigned int *)(v8 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8 * result) = a4;
  ++*(_DWORD *)(v8 + 40);
  return result;
}
