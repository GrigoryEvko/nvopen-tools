// Function: sub_E83380
// Address: 0xe83380
//
__int64 __fastcall sub_E83380(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 296) + 8LL) + 208LL);
  if ( (void (*)())result != nullsub_330 )
    result = ((__int64 (*)(void))result)();
  if ( a2 == 1 )
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL);
    *(_BYTE *)(result + 81) = 1;
  }
  return result;
}
