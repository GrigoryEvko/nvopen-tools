// Function: sub_39A6490
// Address: 0x39a6490
//
__int64 __fastcall sub_39A6490(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rbx
  int v8; // r8d
  int v9; // r9d
  __int64 result; // rax
  __int64 v11[3]; // [rsp+8h] [rbp-18h] BYREF

  v11[0] = a2;
  v7 = sub_39A6170(a1 + 336, v11, a3, a4, a5, a6);
  result = *(unsigned int *)(v7 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(v7 + 12) )
  {
    sub_16CD150(v7, (const void *)(v7 + 16), 0, 8, v8, v9);
    result = *(unsigned int *)(v7 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v7 + 8 * result) = a3;
  ++*(_DWORD *)(v7 + 8);
  return result;
}
