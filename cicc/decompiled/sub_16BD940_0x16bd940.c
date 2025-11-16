// Function: sub_16BD940
// Address: 0x16bd940
//
__int64 __fastcall sub_16BD940(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 result; // rax

  v2 = (unsigned int)(1 << a2);
  *(_DWORD *)(a1 + 16) = v2;
  *(_QWORD *)a1 = &unk_49EF310;
  result = (__int64)_libc_calloc((unsigned int)(v2 + 1), 8u);
  if ( !result )
  {
    if ( (_DWORD)v2 == -1 )
    {
      result = sub_13A3880(1u);
    }
    else
    {
      sub_16BD1C0("Allocation failed", 1u);
      result = 0;
    }
  }
  *(_QWORD *)(result + 8 * v2) = -1;
  *(_QWORD *)(a1 + 8) = result;
  *(_DWORD *)(a1 + 20) = 0;
  return result;
}
