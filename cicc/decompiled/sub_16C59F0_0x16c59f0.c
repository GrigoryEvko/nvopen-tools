// Function: sub_16C59F0
// Address: 0x16c59f0
//
__int64 __fastcall sub_16C59F0(__int64 a1, int a2, int a3, __int64 a4, __off_t a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rdx

  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = a3;
  result = sub_16C5970((size_t *)a1, a2, a5, a3);
  *(_DWORD *)a6 = result;
  *(_QWORD *)(a6 + 8) = v8;
  if ( (_DWORD)result )
    *(_QWORD *)(a1 + 8) = 0;
  return result;
}
