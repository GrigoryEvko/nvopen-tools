// Function: sub_E0EF40
// Address: 0xe0ef40
//
__int64 __fastcall sub_E0EF40(__int64 a1)
{
  __int64 result; // rax

  result = a1;
  if ( !*(_BYTE *)(a1 + 32) )
  {
    *(_BYTE *)(a1 + 32) = 1;
    result = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 24) + 24LL))(*(_QWORD *)(a1 + 24));
    *(_BYTE *)(a1 + 32) = 0;
  }
  return result;
}
