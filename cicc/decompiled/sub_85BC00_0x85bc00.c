// Function: sub_85BC00
// Address: 0x85bc00
//
__int64 __fastcall sub_85BC00(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)result == 2 )
  {
    result = *(_QWORD *)(a1 + 88);
    *(_QWORD *)(result + 200) = **(_QWORD **)(a2 + 32);
    *(_QWORD *)(result + 208) = *(_QWORD *)(a2 + 40);
  }
  else if ( (unsigned __int8)result > 2u )
  {
    if ( (_BYTE)result != 3 )
      sub_721090();
  }
  else
  {
    result = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a1 + 88) = result;
  }
  return result;
}
