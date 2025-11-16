// Function: sub_21E6940
// Address: 0x21e6940
//
__int64 __fastcall sub_21E6940(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 result; // rax

  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8) )
  {
    v5 = *(_QWORD *)(a3 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v5) <= 8 )
    {
      return sub_16E7EE0(a3, ".shiftamt", 9u);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 116;
      *(_QWORD *)v5 = 0x6D6174666968732ELL;
      *(_QWORD *)(a3 + 24) += 9LL;
      return 0x6D6174666968732ELL;
    }
  }
  return result;
}
