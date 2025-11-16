// Function: sub_1636A90
// Address: 0x1636a90
//
__int64 __fastcall sub_1636A90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // [rsp+8h] [rbp-18h]

  result = a1;
  v3 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    result = a1;
    v3 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)result + 8 * v3) = a2;
  v4 = *(unsigned int *)(result + 88);
  ++*(_DWORD *)(result + 8);
  if ( (unsigned int)v4 >= *(_DWORD *)(result + 92) )
  {
    v5 = result;
    sub_16CD150(result + 80, result + 96, 0, 8);
    result = v5;
    v4 = *(unsigned int *)(v5 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(result + 80) + 8 * v4) = a2;
  ++*(_DWORD *)(result + 88);
  return result;
}
