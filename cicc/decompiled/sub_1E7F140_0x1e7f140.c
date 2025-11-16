// Function: sub_1E7F140
// Address: 0x1e7f140
//
__int64 __fastcall sub_1E7F140(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax

  *(_QWORD *)(a1 + 232) = 0;
  *(_DWORD *)(a1 + 560) = 0;
  v2 = *(_QWORD *)(a1 + 616);
  if ( v2 )
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL))(v2);
  *(_QWORD *)(a1 + 616) = 0;
  return result;
}
