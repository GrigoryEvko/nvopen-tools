// Function: sub_1644060
// Address: 0x1644060
//
__int64 __fastcall sub_1644060(__int64 a1, const void *a2, size_t a3)
{
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-28h]

  result = sub_145CBF0((__int64 *)(*(_QWORD *)a1 + 2272LL), 32, 16);
  *(_QWORD *)result = a1;
  *(_QWORD *)(result + 8) = 13;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  if ( a3 )
  {
    v5 = result;
    sub_1643660((__int64 **)result, a2, a3);
    return v5;
  }
  return result;
}
