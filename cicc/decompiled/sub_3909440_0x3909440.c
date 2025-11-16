// Function: sub_3909440
// Address: 0x3909440
//
__int64 __fastcall sub_3909440(__int64 a1, __int64 a2)
{
  *(_QWORD *)(a1 + 8) = a2;
  return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 16LL))(a2, a1);
}
