// Function: sub_30CACB0
// Address: 0x30cacb0
//
__int64 __fastcall sub_30CACB0(__int64 a1)
{
  *(_BYTE *)(a1 + 57) = 1;
  sub_30CAC80((_QWORD *)a1);
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
}
