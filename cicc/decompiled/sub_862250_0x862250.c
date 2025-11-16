// Function: sub_862250
// Address: 0x862250
//
__int64 __fastcall sub_862250(__int64 a1)
{
  __int64 i; // rbx
  __int64 v2; // r12
  __int64 result; // rax

  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
    {
      v2 = *(_QWORD *)(i + 128);
      sub_861D10(v2, *(_BYTE *)(v2 + 28), *(__int64 **)(*(_QWORD *)i + 96LL), 1, 0, 0);
      result = sub_862250(v2);
    }
  }
  return result;
}
