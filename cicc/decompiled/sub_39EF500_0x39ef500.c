// Function: sub_39EF500
// Address: 0x39ef500
//
__int64 __fastcall sub_39EF500(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 8LL) + 136LL);
  if ( (void (*)())result != nullsub_1967 )
    result = ((__int64 (*)(void))result)();
  if ( a2 == 1 )
  {
    result = *(_QWORD *)(a1 + 264);
    *(_BYTE *)(result + 484) |= 2u;
  }
  return result;
}
