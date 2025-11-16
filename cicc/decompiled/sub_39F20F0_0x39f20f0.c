// Function: sub_39F20F0
// Address: 0x39f20f0
//
__int64 __fastcall sub_39F20F0(__int64 a1, int a2)
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
