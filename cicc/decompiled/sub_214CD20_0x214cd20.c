// Function: sub_214CD20
// Address: 0x214cd20
//
__int64 __fastcall sub_214CD20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax

  v7 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
    v7 = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v7) = *(_DWORD *)(a1 + 160);
  v8 = *(unsigned int *)(a1 + 72);
  ++*(_DWORD *)(a1 + 40);
  if ( (unsigned int)v8 >= *(_DWORD *)(a1 + 76) )
  {
    sub_16CD150(a1 + 64, (const void *)(a1 + 80), 0, 8, a5, a6);
    v8 = *(unsigned int *)(a1 + 72);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v8) = a2;
  result = *(unsigned int *)(a1 + 120);
  ++*(_DWORD *)(a1 + 72);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 124) )
  {
    sub_16CD150(a1 + 112, (const void *)(a1 + 128), 0, 8, a5, a6);
    result = *(unsigned int *)(a1 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8 * result) = a3;
  ++*(_DWORD *)(a1 + 120);
  ++*(_DWORD *)a1;
  return result;
}
