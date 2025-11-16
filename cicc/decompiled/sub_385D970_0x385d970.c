// Function: sub_385D970
// Address: 0x385d970
//
__int64 __fastcall sub_385D970(__int64 a1)
{
  __int64 result; // rax

  result = a1;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 16) - 60) <= 0xCu && *(_BYTE *)(**(_QWORD **)(a1 - 24) + 8LL) == 11 )
    return *(_QWORD *)(a1 - 24);
  return result;
}
