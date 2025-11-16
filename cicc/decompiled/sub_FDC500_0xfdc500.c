// Function: sub_FDC500
// Address: 0xfdc500
//
__int64 __fastcall sub_FDC500(__int64 *a1)
{
  __int64 v1; // r8
  __int64 result; // rax

  v1 = *a1;
  *a1 = 0;
  if ( v1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v1 + 8LL))(v1);
  return result;
}
