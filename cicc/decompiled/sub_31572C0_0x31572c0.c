// Function: sub_31572C0
// Address: 0x31572c0
//
__int64 __fastcall sub_31572C0(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax

  if ( (unsigned __int8)(a3 - 4) > 7u )
    return *(_QWORD *)(a1 + 32);
  result = *(_QWORD *)(a1 + 48);
  if ( !result )
    return *(_QWORD *)(a1 + 32);
  return result;
}
