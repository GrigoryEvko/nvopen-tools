// Function: sub_16A1380
// Address: 0x16a1380
//
__int64 __fastcall sub_16A1380(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_14A9E40(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8));
  if ( (_DWORD)result == 1 )
    return sub_14A9E40(*(_QWORD *)(a1 + 8) + 32LL, *(_QWORD *)(a2 + 8) + 32LL);
  return result;
}
