// Function: sub_34BE230
// Address: 0x34be230
//
__int64 __fastcall sub_34BE230(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-18h]

  result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 56LL))(*(_QWORD *)(a1 + 16));
  if ( !BYTE4(result) )
  {
    if ( a2 > 1 )
      BUG();
    BYTE4(v3) = 0;
    return v3;
  }
  return result;
}
