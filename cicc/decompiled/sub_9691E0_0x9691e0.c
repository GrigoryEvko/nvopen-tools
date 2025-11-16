// Function: sub_9691E0
// Address: 0x9691e0
//
__int64 __fastcall sub_9691E0(__int64 a1, unsigned int a2, __int64 a3, unsigned __int8 a4, char a5)
{
  unsigned __int64 v5; // rdx
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
    return sub_C43690(a1, a3, a4);
  if ( a5 || a4 )
  {
    v5 = a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)a2);
    if ( !a2 )
      v5 = 0;
    *(_QWORD *)a1 = v5;
  }
  else
  {
    *(_QWORD *)a1 = a3;
  }
  return result;
}
