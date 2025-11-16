// Function: sub_1F4D850
// Address: 0x1f4d850
//
__int64 __fastcall sub_1F4D850(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, unsigned __int64 *a5)
{
  __int64 *v6; // rdi
  __int64 result; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax

  if ( a5 != (unsigned __int64 *)a4 && a5 != (unsigned __int64 *)a2 )
  {
    v6 = (__int64 *)(a1 + 16);
    if ( v6 != (__int64 *)(a3 + 16) )
      result = sub_1DD5C00(v6, a3 + 16, (__int64)a4, (__int64)a5);
    if ( a5 != (unsigned __int64 *)a2 && a5 != (unsigned __int64 *)a4 )
    {
      v9 = *a5 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 8) = a5;
      *a5 = *a5 & 7 | *a4 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = *a2;
      *(_QWORD *)(v9 + 8) = a2;
      v10 &= 0xFFFFFFFFFFFFFFF8LL;
      *a4 = v10 | *a4 & 7;
      *(_QWORD *)(v10 + 8) = a4;
      result = v9 | *a2 & 7;
      *a2 = result;
    }
  }
  return result;
}
