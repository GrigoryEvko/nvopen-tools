// Function: sub_21E7F80
// Address: 0x21e7f80
//
unsigned __int64 __fastcall sub_21E7F80(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 result; // rax
  _DWORD *v6; // rdx

  result = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( (_DWORD)result )
  {
    v6 = *(_DWORD **)(a3 + 24);
    result = *(_QWORD *)(a3 + 16) - (_QWORD)v6;
    if ( result <= 3 )
    {
      return sub_16E7EE0(a3, ".sat", 4u);
    }
    else
    {
      *v6 = 1952543534;
      *(_QWORD *)(a3 + 24) += 4LL;
    }
  }
  return result;
}
