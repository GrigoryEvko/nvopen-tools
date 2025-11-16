// Function: sub_2BF0980
// Address: 0x2bf0980
//
__int64 __fastcall sub_2BF0980(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 result; // rax

  *(_QWORD *)(a2 + 96) = a1;
  for ( i = *(_QWORD *)(a1 + 120); a1 + 112 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(i - 24) + 24LL))(i - 24, a2);
  }
  return result;
}
