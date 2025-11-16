// Function: sub_15D3960
// Address: 0x15d3960
//
__int64 __fastcall sub_15D3960(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi

  if ( (unsigned __int8)sub_15CDC00(a1) || (result = *(unsigned int *)(a1 + 16), (_DWORD)result) )
  {
    v4 = *(_QWORD *)a1;
    *(_QWORD *)(v4 + 64) = a2;
    result = sub_15D3930(v4);
    *(_DWORD *)(a1 + 16) = 0;
  }
  return result;
}
