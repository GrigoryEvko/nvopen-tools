// Function: sub_30D1170
// Address: 0x30d1170
//
__int64 __fastcall sub_30D1170(_BYTE *a1)
{
  __int64 result; // rax

  result = (*(__int64 (__fastcall **)(_BYTE *))(*(_QWORD *)a1 + 80LL))(a1);
  a1[456] = 0;
  return result;
}
