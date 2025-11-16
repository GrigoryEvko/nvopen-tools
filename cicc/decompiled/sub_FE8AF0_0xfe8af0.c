// Function: sub_FE8AF0
// Address: 0xfe8af0
//
__int64 __fastcall sub_FE8AF0(__int64 a1, unsigned int *a2, __int64 a3)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8) + 24LL * *a2;
  *(_QWORD *)(result + 16) = a3;
  return result;
}
