// Function: sub_1E6F4E0
// Address: 0x1e6f4e0
//
__int64 __fastcall sub_1E6F4E0(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r8
  __int64 result; // rax

  if ( a2 != a1 + 344 )
  {
    v4 = sub_1F03240(a1 + 2128, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL, a2);
    result = 0;
    if ( v4 )
      return result;
    sub_1F03360(a1 + 2128, a2, *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL);
  }
  if ( ((*(_BYTE *)a3 ^ 6) & 6) != 0 )
    sub_1F01A00(a2, a3, 1);
  else
    sub_1F01A00(a2, a3, *(_DWORD *)(a3 + 8) != 3);
  return 1;
}
