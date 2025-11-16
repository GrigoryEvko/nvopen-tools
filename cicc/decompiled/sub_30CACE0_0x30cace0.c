// Function: sub_30CACE0
// Address: 0x30cace0
//
__int64 __fastcall sub_30CACE0(__int64 a1)
{
  *(_BYTE *)(a1 + 57) = 1;
  sub_30CAC80((_QWORD *)a1);
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
}
