// Function: sub_72E370
// Address: 0x72e370
//
__int64 __fastcall sub_72E370(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rdi
  unsigned int v11; // ebx

  v2 = a1[21];
  result = *(unsigned int *)(v2 + 248);
  if ( !(_DWORD)result )
  {
    v5 = sub_72E220((__int64)a1, a2);
    v10 = *(__int64 **)(v2 + 168);
    v11 = v5;
    if ( v10 )
      v11 = sub_72E120(v10, a2, v6, v7, v8, v9) + v5;
    if ( *a1 )
      v11 += *(_DWORD *)(*(_QWORD *)(*a1 + 96LL) + 168LL);
    result = 1;
    if ( v11 )
      result = v11;
    *(_DWORD *)(v2 + 248) = result;
  }
  return result;
}
