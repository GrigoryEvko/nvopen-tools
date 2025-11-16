// Function: sub_16D1890
// Address: 0x16d1890
//
__int64 __fastcall sub_16D1890(__int64 a1, unsigned int a2)
{
  unsigned int v2; // ebx
  __int64 result; // rax
  __int64 v4; // rdx

  if ( !a2 )
  {
    *(_QWORD *)(a1 + 12) = 0;
    result = (__int64)_libc_calloc(17, 0xCu);
    if ( result )
    {
      v4 = 128;
      v2 = 16;
      goto LABEL_6;
    }
    v2 = 16;
LABEL_8:
    sub_16BD1C0("Allocation failed", 1u);
    result = 0;
    v4 = 8LL * v2;
    goto LABEL_6;
  }
  *(_QWORD *)(a1 + 12) = 0;
  v2 = a2;
  result = (__int64)_libc_calloc(a2 + 1, 0xCu);
  if ( !result )
  {
    if ( a2 == -1 )
    {
      v2 = -1;
      result = sub_13A3880(1u);
      v4 = 0x7FFFFFFF8LL;
      goto LABEL_6;
    }
    goto LABEL_8;
  }
  v4 = 8LL * a2;
LABEL_6:
  *(_QWORD *)(result + v4) = 2;
  *(_DWORD *)(a1 + 8) = v2;
  *(_QWORD *)a1 = result;
  return result;
}
