// Function: sub_823950
// Address: 0x823950
//
__int64 __fastcall sub_823950(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    if ( !*(_BYTE *)(*(_QWORD *)(a1 + 32) + result - 1) )
      *(_QWORD *)(a1 + 16) = result - 1;
  }
  return result;
}
