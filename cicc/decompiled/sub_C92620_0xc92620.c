// Function: sub_C92620
// Address: 0xc92620
//
__int64 __fastcall sub_C92620(__int64 a1, unsigned int a2)
{
  int v2; // ebx
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx

  if ( !a2 )
  {
    *(_QWORD *)(a1 + 12) = 0;
    result = _libc_calloc(17, 12);
    if ( result )
    {
      v8 = 128;
      v2 = 16;
      goto LABEL_6;
    }
LABEL_9:
    sub_C64F00("Allocation failed", 1u);
  }
  *(_QWORD *)(a1 + 12) = 0;
  v2 = a2;
  result = _libc_calloc(a2 + 1, 12);
  if ( !result )
  {
    if ( a2 == -1 )
    {
      v2 = -1;
      result = sub_C65340(1, 12, v4, v5, v6, v7);
      v8 = 0x7FFFFFFF8LL;
      goto LABEL_6;
    }
    goto LABEL_9;
  }
  v8 = 8LL * a2;
LABEL_6:
  *(_QWORD *)(result + v8) = 2;
  *(_DWORD *)(a1 + 8) = v2;
  *(_QWORD *)a1 = result;
  return result;
}
