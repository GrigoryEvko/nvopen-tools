// Function: sub_879260
// Address: 0x879260
//
__int64 __fastcall sub_879260(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a2 + 80);
  if ( (_BYTE)result == 16 )
  {
    a2 = **(_QWORD **)(a2 + 88);
    result = *(unsigned __int8 *)(a2 + 80);
  }
  if ( (_BYTE)result == 24 )
    a2 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a1 + 88) = a2;
  if ( a3 != -1 )
  {
    result = *(unsigned int *)(qword_4F04C68[0] + 776LL * a3);
    *(_DWORD *)(a1 + 40) = result;
  }
  return result;
}
