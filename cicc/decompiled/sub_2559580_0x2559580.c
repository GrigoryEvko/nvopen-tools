// Function: sub_2559580
// Address: 0x2559580
//
unsigned __int64 __fastcall sub_2559580(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rbp
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11[4]; // [rsp-20h] [rbp-20h] BYREF

  result = *(_QWORD *)(a1 + 104);
  if ( result )
  {
    _BitScanReverse64(&result, result);
    result ^= 0x3Fu;
    if ( (_DWORD)result != 63 )
    {
      v11[3] = v4;
      v11[0] = sub_A77A40(a3, 63 - (unsigned __int8)result);
      return sub_25594F0(a4, v11, v7, v8, v9, v10);
    }
  }
  return result;
}
