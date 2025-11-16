// Function: sub_1E40390
// Address: 0x1e40390
//
__int64 __fastcall sub_1E40390(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = *a1;
  *a2 = *a1;
  *a1 = (__int64)a2;
  v3 = a1[2];
  if ( v3 )
    return (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v3 + 24LL))(
             v3,
             a2[1],
             a2[2],
             a2[5],
             a2[3],
             a2[4]);
  return result;
}
