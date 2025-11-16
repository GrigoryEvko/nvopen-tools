// Function: sub_2FFA310
// Address: 0x2ffa310
//
__int64 __fastcall sub_2FFA310(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, unsigned __int64 *a5)
{
  __int64 result; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rax

  if ( a5 != (unsigned __int64 *)a4 && a5 != (unsigned __int64 *)a2 )
  {
    result = sub_2E310C0((__int64 *)(a1 + 40), (__int64 *)(a3 + 40), (__int64)a4, (__int64)a5);
    if ( a5 != (unsigned __int64 *)a4 )
    {
      v8 = *a5 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 8) = a5;
      *a5 = *a5 & 7 | *a4 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = *a2;
      *(_QWORD *)(v8 + 8) = a2;
      v9 &= 0xFFFFFFFFFFFFFFF8LL;
      *a4 = v9 | *a4 & 7;
      *(_QWORD *)(v9 + 8) = a4;
      result = v8 | *a2 & 7;
      *a2 = result;
    }
  }
  return result;
}
