// Function: sub_20FF800
// Address: 0x20ff800
//
__int64 __fastcall sub_20FF800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  _DWORD *v7; // rdi
  __int64 v8; // rbx
  __int64 result; // rax

  v7 = *(_DWORD **)(a1 + 40);
  if ( v7 )
    sub_1F5BB60(v7, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(v8 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(v8 + 12) )
  {
    sub_16CD150(v8, (const void *)(v8 + 16), 0, 4, a5, a6);
    result = *(unsigned int *)(v8 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v8 + 4 * result) = a2;
  ++*(_DWORD *)(v8 + 8);
  return result;
}
