// Function: sub_1F4BEC0
// Address: 0x1f4bec0
//
__int64 __fastcall sub_1F4BEC0(__int64 a1, unsigned int *a2)
{
  __int64 result; // rax

  if ( !sub_1F4B670(a1) )
    return sub_1F4BE80(a1, *a2);
  result = sub_38D7340(a1, *(_QWORD *)(a1 + 176), *(_QWORD *)(a1 + 184) + 8LL, a2);
  if ( (int)result < 0 )
    return 1000;
  return result;
}
