// Function: sub_35F2220
// Address: 0x35f2220
//
unsigned __int64 __fastcall sub_35F2220(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int64 result; // rax
  _DWORD *v5; // rdx

  result = *(unsigned int *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( (_DWORD)result )
  {
    v5 = *(_DWORD **)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - (_QWORD)v5;
    if ( result <= 3 )
    {
      return sub_CB6200(a4, ".sat", 4u);
    }
    else
    {
      *v5 = 1952543534;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  return result;
}
