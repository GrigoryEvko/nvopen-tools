// Function: sub_1D46060
// Address: 0x1d46060
//
__int64 __fastcall sub_1D46060(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 )
    a2 += 8;
  result = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)result == a2 )
    *(_QWORD *)result = *(_QWORD *)(*(_QWORD *)result + 8LL);
  return result;
}
