// Function: sub_1F017D0
// Address: 0x1f017d0
//
__int64 __fastcall sub_1F017D0(__int64 a1, __int64 a2)
{
  __int16 v2; // dx
  __int64 result; // rax

  if ( !a2 )
    return 0;
  v2 = *(_WORD *)(a2 + 24);
  result = 0;
  if ( v2 < 0 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) + ((__int64)~v2 << 6);
  return result;
}
