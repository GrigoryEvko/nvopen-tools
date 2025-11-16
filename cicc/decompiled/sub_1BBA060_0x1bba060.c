// Function: sub_1BBA060
// Address: 0x1bba060
//
_QWORD *__fastcall sub_1BBA060(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdx
  __int64 v5; // rdi
  _QWORD *result; // rax
  _QWORD v7[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( a2 && *(_DWORD *)(a2 + 88) != -1 )
  {
    v3 = *(_QWORD *)(a2 + 8);
    --*(_DWORD *)(a2 + 92);
    if ( (*(_DWORD *)(v3 + 96))-- == 1 )
    {
      v7[1] = v2;
      v5 = *a1;
      v7[0] = *(_QWORD *)(a2 + 8);
      return sub_1BB95B0(v5, (__int64)v7);
    }
  }
  return result;
}
