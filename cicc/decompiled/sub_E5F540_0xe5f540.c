// Function: sub_E5F540
// Address: 0xe5f540
//
__int64 __fastcall sub_E5F540(__int64 a1)
{
  __int64 result; // rax

  sub_E5F0E0(a1);
  result = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 24) + 48LL))(*(_QWORD *)(a1 + 24), a1);
  *(_BYTE *)(a1 + 32) = 0;
  return result;
}
